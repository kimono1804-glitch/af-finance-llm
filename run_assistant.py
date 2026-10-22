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
from common.periods import missing_periods, period_not_available_report, resolve_relative_periods
from common.prompts import (CONTRACT, SYSTEM_PROMPT, adapter_contract, build_tools, build_user_message,
                            parse_tool_calls, split_think, strip_tool_calls)
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

    def __init__(self, adapter_dir: Path | None, base_model: str, max_seq_length: int = 8192,
                 contract: str | None = None):
        """adapter_dir=None loads the base model on purpose (used for base-vs-fine-tuned evaluation).
        contract: prompt contract (tool list) to show the model; default = what the adapter was trained on."""
        from unsloth import FastLanguageModel

        if adapter_dir is not None and (adapter_dir / "adapter_config.json").exists():
            name = str(adapter_dir)
            self.label = f"fine-tuned ({adapter_dir.name})"
            print(f"Loading fine-tuned adapter: {adapter_dir}")
        else:
            name = base_model
            self.label = "base (not fine-tuned)"
            if adapter_dir is not None:
                print(f"WARNING: no adapter at {adapter_dir}; using base model {base_model} (not fine-tuned).")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=name, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(self.model)
        fine_tuned = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
        self.contract = contract or adapter_contract(adapter_dir if fine_tuned else None)
        self.tools = build_tools(self.contract)
        self.max_time: float | None = None      # seconds per generate() call; None = no limit
        self.log_rounds = False                 # print tokens / seconds for every generate() call
        self.last_stats: dict[str, float] = {}

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        import time

        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=True,
                                                    enable_thinking=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20) if temperature > 0 else dict(do_sample=False)
        extra = {"max_time": self.max_time} if self.max_time else {}
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling, **extra,
                                      pad_token_id=self.tokenizer.eos_token_id)
        n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
        dt = time.time() - t0
        self.last_stats = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": n_new, "seconds": dt}
        if self.log_rounds:
            hit = " (TIME LIMIT)" if self.max_time and dt >= self.max_time - 1 else ""
            print(f"    gen: prompt {inputs['input_ids'].shape[1]} tok, +{n_new} tok in {dt:.1f}s "
                  f"({n_new / max(dt, 1e-6):.1f} tok/s){hit}", flush=True)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(tok)[0]
        return text.strip()


class ServerLLM:
    """Any OpenAI-compatible chat server with tool calling: llama.cpp llama-server, Ollama, vLLM, LM Studio.

    Converts our Qwen3-style message list to the OpenAI format and the server's reply back into Qwen3 text
    (<think>...</think> + <tool_call>{...}</tool_call>), so the rest of the pipeline is identical to local mode.
    """

    def __init__(self, url: str, model: str, timeout: float = 600.0, contract: str = CONTRACT):
        self.url = url.rstrip("/") + "/chat/completions"
        self.contract = contract
        self.tools = build_tools(contract)
        self.model = model
        self.timeout = timeout
        self.label = f"server ({model} @ {url})"

    @staticmethod
    def to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending: list[str] = []
        n = 0
        for m in messages:
            if m["role"] == "assistant":
                think, rest = split_think(m.get("content") or "")
                msg: dict[str, Any] = {"role": "assistant", "content": rest}
                if think:
                    msg["reasoning_content"] = think
                if m.get("tool_calls"):
                    msg["tool_calls"] = []
                    for tc in m["tool_calls"]:
                        n += 1
                        cid = f"call_{n}"
                        pending.append(cid)
                        fn = tc["function"]
                        args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                        msg["tool_calls"].append({"id": cid, "type": "function",
                                                  "function": {"name": fn["name"], "arguments": args}})
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_0",
                            "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        import urllib.request

        body = {"model": self.model, "messages": self.to_openai(messages), "tools": self.tools,
                "max_tokens": max_new_tokens, "temperature": temperature, "top_p": 0.95}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
        content = msg.get("content") or ""
        think = msg.get("reasoning_content") or msg.get("reasoning") or ""
        text = content if (not think or "<think>" in content) else f"<think>\n{think}\n</think>\n\n{content}"
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                text += "\n<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": args}) + "\n</tool_call>"
            except json.JSONDecodeError:
                text += f"\n<tool_call>\n{{\"name\": \"{fn.get('name')}\", \"arguments\": {raw}\n</tool_call>"
        return text


def period_guard(question: str, known_periods: set[str], sources: set[str] | None = None) -> dict[str, Any] | None:
    """If the question names a month/quarter that is not in the ingested data, answer without the model.

    The v1 model otherwise uses a neighbouring period's column and labels it as the asked period.
    Returns a trace in the run_conversation format, or None when every asked period is present.
    """
    missing = missing_periods(question, known_periods)
    if not missing:
        return None
    return {"thinks": [f"Period guard: {', '.join(missing)} not in the data; model not called."], "calls": [],
            "report": period_not_available_report(missing, known_periods, sources or ()),
            "guard": "period_not_available"}


def answer(llm: FinanceLLM, context_chunks: list[str], question: str, max_new_tokens: int, temperature: float,
           max_rounds: int = 3) -> dict[str, Any]:
    return run_conversation(llm, build_user_message(context_chunks, question), max_new_tokens, temperature, max_rounds)


def run_conversation(llm: FinanceLLM, user: str, max_new_tokens: int, temperature: float,
                     max_rounds: int = 3) -> dict[str, Any]:
    """Generate -> execute tool calls -> generate, until the model writes a report without a tool call.

    Returns {"thinks": [...], "calls": [{"call", "result", "unsupported_inputs"}], "report": str}.
    """
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
    ap.add_argument("--backend", choices=["local", "server"], default="local",
                    help="local = Unsloth + adapter on this GPU; server = OpenAI-compatible server (see exports/SERVING.md)")
    ap.add_argument("--server-url", default="http://localhost:8080/v1")
    ap.add_argument("--server-model", default="af-finance-qwen3-4b")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--contract", choices=["v1", "v2"], default=None,
                    help="prompt contract (tool list); default = the one the configured adapter was trained on")
    ap.add_argument("--show-context", action="store_true", help="print full retrieved chunk text")
    args = ap.parse_args()
    from common.winperf import boost_process

    boost_process()  # keep the GPU-feeding thread off Windows efficiency cores

    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    chunks = ingest_paths(args.files, retriever.cfg["max_rows_per_chunk"])
    retriever.index(chunks)
    print(f"Indexed {len(chunks)} chunks from {', '.join(Path(f).name for f in args.files)}")

    llm = None
    if not args.retrieval_only and args.backend == "server":
        llm = ServerLLM(args.server_url, args.server_model,
                        contract=args.contract or adapter_contract(ROOT / train_cfg["adapter_dir"]))
    elif not args.retrieval_only:
        import torch

        if not torch.cuda.is_available():
            sys.exit("CUDA GPU not available; use --retrieval-only or install CUDA torch.")
        torch.manual_seed(train_cfg["seed"])
        adapter = Path(args.adapter) if args.adapter else ROOT / train_cfg["adapter_dir"]
        llm = FinanceLLM(adapter, train_cfg["model_name"], contract=args.contract)

    known_periods = {p for c in chunks for p in c.metadata.periods}
    for asked in args.question or DEMO_QUESTIONS:
        q, notes = resolve_relative_periods(asked, known_periods)
        for note in notes:
            print(f"NOTE: {note}")
        guarded = period_guard(q, known_periods, {c.metadata.source for c in chunks})
        if guarded is not None:
            print_trace(q, [], guarded, 0.0)
            continue
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
