"""End-to-end CFO assistant: ingest MIS -> hybrid retrieval -> fine-tuned Qwen3 -> calculator tool -> CFO report.

For each question it prints:
  1. the retrieved chunks (with BM25 / dense ranks and rerank score)
  2. the <think> audit trace(s) from the model
  3. every finance_calculator call, its result, and a faithfulness check (all input numbers must
     appear in the retrieved context)
  4. the final structured CFO report

Usage:
  python run_assistant.py                                   # sample MIS + demo questions
  python run_assistant.py --question "What is CM2 for Feb-26?"
  python run_assistant.py --files data/private/our_mis.xlsx --question "..."
  python run_assistant.py --retrieval-only                  # no LLM: just show what would be retrieved
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from common.audit import unsupported_inputs
from common.prompts import SYSTEM_PROMPT, TOOLS, build_user_message, parse_tool_calls, split_think, strip_tool_calls
from common.tools import run_tool_call
from ingest_financials import ingest_paths
from rag_pipeline import HybridRetriever

ROOT = Path(__file__).resolve().parent

DEMO_QUESTIONS = [
    "What were CM1 and CM2 for Mar-26, in absolute terms and as % of net revenue?",
    "What is our LTV:CAC for Jan-26?",
    "Reconcile EBITDA for Mar-26 from the P&L.",
    "If we make the planned hires, how does our monthly break-even change?",
]

RULE = "=" * 100


class FinanceLLM:
    """Qwen3 + LoRA adapter (falls back to the base model if no adapter is found)."""

    def __init__(self, adapter_dir: Path, base_model: str, max_seq_length: int = 8192):
        from unsloth import FastLanguageModel

        if (adapter_dir / "adapter_config.json").exists():
            name = str(adapter_dir)
            print(f"Loading fine-tuned adapter: {adapter_dir}")
        else:
            name = base_model
            print(f"WARNING: no adapter at {adapter_dir}; using base model {base_model} (not fine-tuned).")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=name, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(self.model)

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                                    enable_thinking=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20) if temperature > 0 else dict(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(tok)[0]
        return text.strip()


def answer(llm: FinanceLLM, context_chunks: list[str], question: str, max_new_tokens: int, temperature: float,
           max_rounds: int = 3) -> dict[str, Any]:
    user = build_user_message(context_chunks, question)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    trace: dict[str, Any] = {"thinks": [], "calls": [], "report": ""}
    for _ in range(max_rounds):
        raw = llm.generate(messages, max_new_tokens, temperature)
        think, rest = split_think(raw)
        trace["thinks"].append(think)
        calls = parse_tool_calls(rest)
        if not calls:
            trace["report"] = strip_tool_calls(rest)
            return trace
        messages.append({"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n",
                         "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls if "error" not in c]})
        for c in calls:
            result = (json.dumps({"results": {}, "errors": {"_call": c["error"]}}) if "error" in c
                      else run_tool_call(c["name"], c["arguments"]))
            unsupported = unsupported_inputs(c["arguments"], user) if isinstance(c["arguments"], dict) else []
            trace["calls"].append({"call": c, "result": json.loads(result), "unsupported_inputs": unsupported})
            messages.append({"role": "tool", "content": result})
    trace["report"] = "(stopped: too many tool-call rounds without a final report)"
    return trace


def print_trace(question: str, hits, trace: dict[str, Any], elapsed: float) -> None:
    print(f"\n{RULE}\nQUESTION: {question}\n{RULE}")
    print("\n--- Retrieved context (hybrid BM25 + dense -> reranked) ---")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.chunk.title:<60} rerank={h.score:7.3f}  bm25#{h.bm25_rank}  dense#{h.dense_rank}")
    for n, think in enumerate(trace["thinks"], 1):
        print(f"\n--- <think> audit trace (turn {n}) ---\n{think or '(empty)'}")
    for c in trace["calls"]:
        print("\n--- finance_calculator call ---")
        print(json.dumps(c["call"].get("arguments"), indent=1))
        print("--- result ---")
        for k, r in c["result"].get("results", {}).items():
            print(f"  {k:<28} {r['display']}")
        for k, e in c["result"].get("errors", {}).items():
            print(f"  {k:<28} ERROR: {e}")
        if c["unsupported_inputs"]:
            print(f"  FAITHFULNESS WARNING: inputs not found in retrieved context: {c['unsupported_inputs']}")
        else:
            print("  Faithfulness: every input number appears in the retrieved context.")
    if not trace["calls"]:
        print("\n(no calculator call: the model answered directly or reported missing data)")
    print(f"\n--- Final CFO report ({elapsed:.0f}s) ---\n{trace['report']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=[str(ROOT / "sample_startup_mis.xlsx")])
    ap.add_argument("--question", action="append", help="repeatable; defaults to demo questions")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default from configs/train.yaml)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6, help="Qwen3 thinking-mode default; 0 = greedy")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--show-context", action="store_true", help="print full retrieved chunk text")
    args = ap.parse_args()

    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    chunks = ingest_paths(args.files, retriever.cfg["max_rows_per_chunk"])
    retriever.index(chunks)
    print(f"Indexed {len(chunks)} chunks from {', '.join(Path(f).name for f in args.files)}")

    llm = None
    if not args.retrieval_only:
        import torch

        if not torch.cuda.is_available():
            sys.exit("CUDA GPU not available; use --retrieval-only or install CUDA torch.")
        torch.manual_seed(train_cfg["seed"])
        adapter = Path(args.adapter) if args.adapter else ROOT / train_cfg["adapter_dir"]
        llm = FinanceLLM(adapter, train_cfg["model_name"])

    for q in args.question or DEMO_QUESTIONS:
        hits = retriever.search(q, args.top_k)
        context = [h.chunk.text for h in hits]
        if args.show_context or llm is None:
            print(f"\n{RULE}\nQUESTION: {q}\n{RULE}")
            for i, h in enumerate(hits, 1):
                print(f"\n[{i}] rerank={h.score:.3f} bm25#{h.bm25_rank} dense#{h.dense_rank}\n{h.chunk.text}")
        if llm is None:
            continue
        t0 = time.time()
        trace = answer(llm, context, q, args.max_new_tokens, args.temperature)
        print_trace(q, hits, trace, time.time() - t0)


if __name__ == "__main__":
    main()
