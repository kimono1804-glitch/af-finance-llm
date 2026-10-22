"""Evaluate the finance model and write a Markdown report to reports/.

Suites
  synthetic  Fresh RAFT examples from generate_raft_data with a different seed (configs/eval.yaml):
             n per task family + missing-data cases. The model gets the gold context ("oracle context"),
             so this measures reasoning/format/tool use, not retrieval.
  realworld  Natural questions about sample_startup_mis.xlsx (eval_cases.py). Context comes from the
             hybrid retriever, so this measures the full RAG + model system. Also reports retrieval recall.

Models
  ft    fine-tuned adapter (configs/train.yaml adapter_dir)
  base  base Qwen3-4B with the same prompt and tools (no fine-tuning)
  On realworld, base = "RAG only" and ft = "RAG + fine-tune".

Scores (common/scoring.py): numeric accuracy vs Python ground truth, value recall, formula accuracy,
context faithfulness, report consistency, end-to-end, missing-data handling, edge-case handling,
false refusals, latency.

Usage
  python eval_model.py                              # both suites, ft + base
  python eval_model.py --models ft --n-per-task 5   # quick check
  python eval_model.py --gold                       # self-test: score the gold answers (must be 100%)
  python eval_model.py --retrieval-only             # realworld retrieval recall only, no LLM
Raw per-case results go to outputs/eval_runs/<run>.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from common.prompts import build_user_message
from common.scoring import (METRICS, Gold, aggregate, aggregate_by, fmt_pct, required_formulas_from_steps, score,
                            trace_from_gold_messages)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Building the cases
# ---------------------------------------------------------------------------

def synthetic_cases(cfg: dict[str, Any], v1_compat: bool = False) -> list[tuple[Gold, str, list[dict]]]:
    """(gold, user_message, gold_messages) for fresh examples from a seed the training set never used."""
    from generate_raft_data import TASKS, build_example

    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    data_cfg["seed"] = cfg["synthetic"]["seed"]
    if v1_compat:
        data_cfg["compat_v1"] = True
    every = cfg["synthetic"]["missing_every"]
    out = []
    for t_i, task in enumerate(cfg.get("tasks") or list(TASKS)):
        for i in range(cfg["synthetic"]["n_per_task"]):
            idx = 1_000_000 + t_i * 10_000 + i
            ex = build_example(idx, data_cfg, task, missing=(i % every == every - 1), split="eval")
            meta, msgs = ex["meta"], ex["messages"]
            truth = {k: (None if v is None else [v]) for k, v in (meta.get("truth") or {}).items()}
            steps = msgs[2]["tool_calls"][0]["function"]["arguments"]["steps"] if "tool_calls" in msgs[2] else []
            question = msgs[1]["content"].rsplit("Question:", 1)[-1].strip()
            missing_kw = [meta["missing"].split("(")[0].strip().rstrip("%").strip()] if meta["missing"] else []
            gold = Gold(case_id=meta["id"], suite="synthetic", task=task, question=question,
                        context=msgs[1]["content"], truth=truth, required_formulas=required_formulas_from_steps(steps),
                        missing=meta["missing"], missing_keywords=missing_kw, edge_cases=meta["edge_cases"])
            out.append((gold, msgs[1]["content"], msgs))
    return out


def realworld_cases(top_k: int | None, period_guard: bool = True,
                    v1_compat: bool = False) -> tuple[list[tuple[Gold, str, Any]], list[dict[str, Any]]]:
    """Retrieve context for every real-world question (CPU) and record retrieval recall.

    With period_guard, a question about a period that is not in the MIS gets the deterministic
    run_assistant.period_guard answer (third tuple element) instead of a model call, as in the real assistant.
    """
    from eval_cases import V1_CASE_IDS, build_cases, build_lakh_cases
    from make_sample_mis import build_workbook
    from ingest_financials import ingest_paths
    from rag_pipeline import HybridRetriever
    from run_assistant import period_guard as guard_fn

    from common.periods import resolve_relative_periods

    retriever = HybridRetriever()
    groups = [(ROOT / "sample_startup_mis.xlsx", build_cases())]
    if not v1_compat:  # lakh-format copy of the same company (added 2026-10-22), built fresh each run
        lakh_path = ROOT / "outputs" / "sample_startup_mis_lakh.xlsx"
        lakh_path.parent.mkdir(exist_ok=True)
        groups.append((build_workbook(lakh_path, lakh=True), build_lakh_cases()))
    out, recall_rows = [], []
    for path, group_cases in groups:
        chunks = ingest_paths([path], retriever.cfg["max_rows_per_chunk"])
        retriever.index(chunks)
        periods = {p for c in chunks for p in c.metadata.periods}
        out_g, recall_g = _realworld_group(retriever, chunks, periods, group_cases, top_k, period_guard, v1_compat,
                                           V1_CASE_IDS, guard_fn, resolve_relative_periods)
        out += out_g
        recall_rows += recall_g
    del retriever
    gc.collect()
    return out, recall_rows


def _realworld_group(retriever, chunks, periods, cases, top_k, period_guard, v1_compat, v1_ids, guard_fn,
                     resolve_relative_periods) -> tuple[list[tuple[Gold, str, Any]], list[dict[str, Any]]]:
    out, recall_rows = [], []
    for case in cases:
        if v1_compat and case.case_id not in v1_ids:
            continue
        question, _ = resolve_relative_periods(case.question, periods)  # same preprocessing as run_assistant
        hits = retriever.search(question, top_k)
        sections = {h.chunk.metadata.section for h in hits}
        found = case.needed_sections & sections
        recall_rows.append({"case_id": case.case_id, "task": case.task, "needed": sorted(case.needed_sections),
                            "retrieved": [h.chunk.metadata.section for h in hits],
                            "recall": len(found) / len(case.needed_sections) if case.needed_sections else 1.0})
        user = build_user_message([h.chunk.text for h in hits], question)
        gold = Gold(case_id=case.case_id, suite="realworld", task=case.task, question=case.question, context=user,
                    truth=case.truth, required_formulas=case.required_formulas, missing=case.missing,
                    missing_keywords=case.missing_keywords, edge_cases=case.edge_cases)
        guarded = guard_fn(question, periods, {c.metadata.source for c in chunks}) if period_guard else None
        out.append((gold, user, guarded))
    return out, recall_rows


# ---------------------------------------------------------------------------
# Running a model
# ---------------------------------------------------------------------------

def run_model(name: str, make_llm: Any, cases: list[tuple[Gold, str, Any]],
              out_path: Path, max_new_tokens: int, temperature: float,
              max_time: float | None = None) -> list[dict[str, Any]]:
    from run_assistant import run_conversation

    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["case_id"]] = r
    todo = [c for c in cases if c[0].case_id not in done]
    print(f"[{name}] {len(done)} cached, {len(todo)} to run -> {out_path}")
    if todo:
        llm = make_llm()
        if hasattr(llm, "max_time"):  # local model: cap every generate() call and log its speed
            llm.max_time = float(max_time) if max_time else None
            llm.log_rounds = True
        with out_path.open("a", encoding="utf-8") as f:
            for n, (gold, user, extra) in enumerate(todo, 1):
                t0 = time.time()
                try:
                    if isinstance(extra, dict) and extra.get("guard"):  # answered by the period guard, no model call
                        trace = extra
                    else:
                        trace = run_conversation(llm, user, max_new_tokens, temperature)
                except Exception as exc:  # noqa: BLE001 - one bad case must not kill a multi-hour run
                    trace = {"thinks": [], "calls": [], "report": f"(generation error: {exc})"}
                row = score(trace, gold)
                row["latency_s"] = round(time.time() - t0, 1)
                row["trace"] = trace
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[gold.case_id] = row
                ok = "OK " if row["end_to_end"] else "FAIL"
                print(f"[{name}] {n}/{len(todo)} {ok} {gold.case_id} ({gold.task}) {row['latency_s']}s"
                      + ("" if row["end_to_end"] else f"  - {row.get('why', '')}"))
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    ids = [c[0].case_id for c in cases]
    return [done[i] for i in ids if i in done]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def metrics_table(runs: dict[str, dict[str, Any]]) -> list[str]:
    names = list(runs)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, _ in METRICS:
        lines.append(f"| {label} | " + " | ".join(fmt_pct(runs[n][key]) for n in names) + " |")
    lines.append("| Cases (answerable / missing) | " + " | ".join(
        f"{runs[n]['n_answerable']} / {runs[n]['n_missing']}" for n in names) + " |")
    lines.append("| Mean latency per case | " + " | ".join(
        "n/a" if runs[n]["latency_s"] is None else f"{runs[n]['latency_s']:.0f}s" for n in names) + " |")
    return lines


def write_report(path: Path, results: dict[str, list[dict[str, Any]]], recall_rows: list[dict[str, Any]],
                 header: list[str]) -> None:
    lines = ["# Model Evaluation Report", "", *header, ""]
    lines += ["## Metric definitions", ""] + [f"- **{label}**: {desc}." for _, label, desc in METRICS] + [""]
    for suite in ("synthetic", "realworld"):
        runs = {name: aggregate([r for r in rows if r["suite"] == suite]) for name, rows in results.items()
                if any(r["suite"] == suite for r in rows)}
        if not runs:
            continue
        title = ("Synthetic held-out (oracle context)" if suite == "synthetic"
                 else "Real-world questions on sample MIS (hybrid RAG context)")
        lines += [f"## {title}", "", *metrics_table(runs), ""]
        for name, rows in results.items():
            srows = [r for r in rows if r["suite"] == suite]
            if not srows:
                continue
            by_task = aggregate_by(srows, "task")
            lines += [f"### {name}: by task", "", "| Task | Cases | Numeric acc. | Formula acc. | Faithful | Missing handled | End-to-end |",
                      "|---|---|---|---|---|---|---|"]
            for task, a in by_task.items():
                lines.append(f"| {task} | {a['n']} | {fmt_pct(a['numeric_accuracy'])} | {fmt_pct(a['formula_accuracy'])} | "
                             f"{fmt_pct(a['faithfulness'])} | {fmt_pct(a['missing_handling'])} | {fmt_pct(a['end_to_end'])} |")
            fails = [r for r in srows if not r["end_to_end"]][:8]
            if fails:
                lines += ["", f"Failures ({name}, first {len(fails)}):", ""]
                lines += [f"- `{r['case_id']}` {r['task']}: {r.get('why', '')} - \"{r['question'][:90]}\"" for r in fails]
            lines.append("")
    if recall_rows:
        mean = sum(r["recall"] for r in recall_rows) / len(recall_rows)
        lines += ["## Retrieval recall (real-world, top-k chunks)", "", f"Mean recall of needed sections: **{fmt_pct(mean)}**", "",
                  "| Case | Task | Needed | Recall | Retrieved (in order) |", "|---|---|---|---|---|"]
        for r in recall_rows:
            lines.append(f"| {r['case_id']} | {r['task']} | {', '.join(r['needed'])} | {fmt_pct(r['recall'])} | "
                         f"{', '.join(r['retrieved'])} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {path}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "eval.yaml"))
    ap.add_argument("--suite", choices=["synthetic", "realworld", "all"], default="all")
    ap.add_argument("--models", choices=["ft", "base", "both"], default="both")
    ap.add_argument("--n-per-task", type=int, help="override synthetic n_per_task")
    ap.add_argument("--gold", action="store_true", help="score the gold answers (scorer self-test, no LLM)")
    ap.add_argument("--retrieval-only", action="store_true", help="real-world retrieval recall only, no LLM")
    ap.add_argument("--backend", choices=["local", "server"], default="local",
                    help="server = evaluate an exported model behind an OpenAI-compatible server (e.g. GGUF in llama-server)")
    ap.add_argument("--server-url", default="http://localhost:8080/v1")
    ap.add_argument("--server-model", default="af-finance-qwen3-4b")
    ap.add_argument("--tag", default="", help="suffix for run files and report name")
    ap.add_argument("--report", help="report path (default reports/eval_<date>[_tag].md)")
    ap.add_argument("--no-period-guard", action="store_true",
                    help="real-world suite: send questions about absent periods to the model (measures the raw model)")
    ap.add_argument("--adapter", help="adapter dir for the fine-tuned model (default: configs/train.yaml adapter_dir)")
    ap.add_argument("--contract", choices=["v1", "v2"], default=None,
                    help="prompt contract for every model; default = the configured adapter's (base gets the same)")
    ap.add_argument("--v1-compat", action="store_true",
                    help="rebuild the cases exactly as the v1 run started them (2026-10-21, before 22:55): no "
                         "row-decimal alignment, no period-gap missing cases, no period guard")
    args = ap.parse_args()
    if args.v1_compat:
        import common.tables

        common.tables.ALIGN_ROW_DECIMALS = False
        args.no_period_guard = True
    from common.winperf import boost_process

    boost_process()  # keep the GPU-feeding thread off Windows efficiency cores

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.n_per_task:
        cfg["synthetic"]["n_per_task"] = args.n_per_task
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    adapter = Path(args.adapter) if args.adapter else ROOT / train_cfg["adapter_dir"]
    adapter_arg = adapter.name
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    tag = f"_{args.tag}" if args.tag else ""
    report = Path(args.report) if args.report else ROOT / "reports" / f"eval_{stamp}{tag}.md"

    cases: list[tuple[Gold, str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    if args.suite in ("synthetic", "all"):
        cases += synthetic_cases(cfg, v1_compat=args.v1_compat)
    if args.suite in ("realworld", "all") and not args.gold:
        rw, recall_rows = realworld_cases(cfg.get("top_k"), period_guard=not args.no_period_guard,
                                         v1_compat=args.v1_compat)
        cases += rw
    print(f"{len(cases)} cases ({sum(1 for c in cases if c[0].missing)} missing-data)")

    header = [f"- Date: {dt.datetime.now():%Y-%m-%d %H:%M}",
              f"- Base model: `{train_cfg['model_name']}`; adapter: `{adapter_arg}`",
              f"- Synthetic: seed {cfg['synthetic']['seed']}, {cfg['synthetic']['n_per_task']} per task family "
              f"(every {cfg['synthetic']['missing_every']}th is a missing-data case)",
              f"- Decoding: temperature {cfg['temperature']}, max_new_tokens {cfg['max_new_tokens']}",
              f"- Real-world period guard: {'off (raw model)' if args.no_period_guard else 'on (absent periods answered without the model)'}"]

    if args.retrieval_only:
        write_report(report, {}, recall_rows, header + ["- Mode: retrieval only (no LLM)"])
        return
    if args.gold:
        rows = [score(trace_from_gold_messages(msgs), gold) for gold, _, msgs in cases]
        write_report(report, {"gold answers": rows}, [], header + ["- Mode: scorer self-test on gold answers"])
        a = aggregate(rows)
        print({k: fmt_pct(a[k]) for k, _, _ in METRICS})
        bad = [r for r in rows if not r["end_to_end"]]
        for r in bad[:10]:
            print("GOLD FAIL", r["case_id"], r.get("why"))
        sys.exit(1 if bad else 0)

    runs_dir = ROOT / "outputs" / "eval_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    from common.prompts import adapter_contract

    contract = args.contract or ("v1" if args.v1_compat else adapter_contract(adapter))
    header.append(f"- Prompt contract: {contract}")
    if args.backend == "server":
        from run_assistant import ServerLLM

        name = f"server: {args.server_model}"
        results = {name: run_model(name, lambda: ServerLLM(args.server_url, args.server_model, contract=contract), cases,
                                   runs_dir / f"server_{args.server_model.replace(':', '_')}{tag}.jsonl",
                                   cfg["max_new_tokens"], cfg["temperature"], cfg.get("max_time_per_turn"))}
        write_report(report, results, recall_rows, header + [f"- Backend: {args.server_url}"])
        return
    plan = {"ft": [("fine-tuned", adapter)], "base": [("base", None)],
            "both": [("fine-tuned", adapter), ("base", None)]}[args.models]
    if any(a is not None for _, a in plan) and not (adapter / "adapter_config.json").exists():
        sys.exit(f"No adapter at {adapter}. Train first, or use --models base.")
    results: dict[str, list[dict[str, Any]]] = {}
    from run_assistant import FinanceLLM

    for name, ad in plan:
        def make_llm(ad: Path | None = ad) -> Any:
            import torch

            torch.manual_seed(0)
            return FinanceLLM(ad, train_cfg["model_name"], contract=contract)

        results[name] = run_model(name, make_llm, cases, runs_dir / f"{name.replace('-', '_')}{tag}.jsonl",
                                  cfg["max_new_tokens"], cfg["temperature"], cfg.get("max_time_per_turn"))
    write_report(report, results, recall_rows, header)


if __name__ == "__main__":
    main()
