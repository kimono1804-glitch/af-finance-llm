# af-finance-llm

A startup CFO assistant that reads your MIS / board-pack tables (Excel or PDF) and answers finance questions (net revenue, gross margin, CM1-CM3, EBITDA, break-even, LTV:CAC, payback, runway, burn multiple, magic number) without making up numbers.

| Layer | Does | Code |
|---|---|---|
| Fine-tuned Qwen3-4B (QLoRA) | Reads the context, picks the period and line items, chooses the formula, writes the CFO report. Never does arithmetic. | `train_qwen3.py` |
| Table-aware hybrid RAG | Finds the right P&L / unit-economics rows. Every chunk keeps its headers, periods and source. | `ingest_financials.py`, `rag_pipeline.py` |
| Python calculator tool | Computes every number from `common/formulas.py`. | `common/tools.py` |

```
question ─► hybrid retrieval (BM25 + bge-large + RRF + bge-reranker) ─► top-5 table chunks
        ─► Qwen3 <think> (extract inputs, pick formulas) ─► finance_calculator tool call
        ─► Python computes ─► Qwen3 writes the CFO report using the tool's exact values
```

If a required input is missing from the context, the model says which one and does not call the tool.

## Setup

Requirements: Windows/Linux, NVIDIA GPU with **8 GB+ VRAM** (developed on an RTX 5050 Laptop, 8 GB), CUDA 12.8+ driver, Python 3.11/3.12.

```bash
conda create -n unsloth_env python=3.12 -y
conda activate unsloth_env
# CUDA build of torch first (cu130 for recent drivers; cu128 also works on RTX 50xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install unsloth trl peft bitsandbytes datasets chromadb pytest pyyaml
pip install -e ".[pdf,qdrant,dev]"
```

The first run downloads `unsloth/Qwen3-4B-unsloth-bnb-4bit` (~3.5 GB), `BAAI/bge-large-en-v1.5` (~1.3 GB) and `BAAI/bge-reranker-large` (~2.2 GB).

## Commands

```bash
# Tests (formulas, calculator, dataset validation, ingestion, BM25 retrieval). No GPU needed.
pytest

# 1. Synthetic RAFT dataset: 1,000 examples (90/10 train/test), validated against Python ground truth
python generate_raft_data.py            # -> dataset.jsonl
python generate_raft_data.py --validate-only

# 2. QLoRA fine-tune (r=32, alpha=32, all linear layers, responses only, packing=False)
# configs/train.yaml adapter_dir is the default adapter for the assistant, eval and export
# (now qwen3_4b_finance_lora_v2). Training refuses to replace an existing adapter: set a new
# adapter_dir in the config, or pass --overwrite.
python train_qwen3.py --max-steps 5     # smoke test
python train_qwen3.py                   # -> adapter_dir (+ loss_curve.png)
# v2 (11 tasks, lakh/crore, multi-period; separate data / checkpoints / adapter; report reports/eval_2026-10-23_v2.md):
python generate_raft_data.py --n 3000 --output dataset_v2.jsonl
python train_qwen3.py --config configs/train_v2.yaml   # -> qwen3_4b_finance_lora_v2/
python eval_model.py --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2

# 3. Ingestion + retrieval
python make_sample_mis.py               # -> sample_startup_mis.xlsx (synthetic company)
python ingest_financials.py sample_startup_mis.xlsx
python rag_pipeline.py sample_startup_mis.xlsx "What were CM1 and CM2 for Mar-26?"

python make_sample_mis.py --pdf sample_board_pack.pdf  # same data as a board-pack PDF (Docling test)

# 4. End-to-end assistant
python run_assistant.py                                  # demo questions on the sample MIS
python run_assistant.py --question "What is our CAC payback for Feb-26?"
python run_assistant.py --question "What was CM2 last month?"   # "last month" = latest month in the data
python run_assistant.py --files data/private/our_mis.xlsx --question "..."
python run_assistant.py --retrieval-only                 # no LLM, just show retrieved chunks

# 5. Evaluation -> reports/eval_<date>.md
python eval_model.py --gold                              # scorer self-test (must be 100%)
python eval_model.py --retrieval-only                    # retrieval recall on real-world questions
python eval_model.py                                     # fine-tuned vs base, synthetic + real-world (hours)
python eval_model.py --models ft --n-per-task 5          # quick check
python eval_model.py --suite realworld --no-period-guard # raw model on absent-period questions (no guard)
python eval_model.py --tag v1 --v1-compat               # resume / repeat the v1 evaluation on identical cases

# 6. Export for serving (exports/, git-ignored)
# Merges the LoRA into the dequantized 4-bit base it was trained on (Unsloth's own merge uses the original
# bf16 weights and lost 14 points), then llama.cpp converts + quantizes. ~3 min on CPU, no GPU needed.
python export.py                                         # exports/merged_16bit/ (vLLM) + exports/gguf/<name>.Q8_0.gguf + Modelfile + SERVING.md
python export.py --gguf q8_0 q6_k                        # other types: evaluate each before shipping (q4_k_m was much worse)
# then serve it (needs a GPU build of llama.cpp for -ngl) and point the assistant (or the evaluator) at the server:
llama-server -m exports/gguf/af-finance-qwen3-4b.Q8_0.gguf --jinja -ngl 99 -c 8192 -np 1 -fa on --port 8080 --alias af-finance-qwen3-4b
python run_assistant.py --backend server --server-url http://localhost:8080/v1 --contract v2
python eval_model.py --backend server --contract v2 --config configs/eval_v2.yaml --tag gguf   # quantized vs adapter
```

Slow tests (Docling PDF ingestion): `pytest -m slow`.

## Data and privacy

- Put real MIS / board packs in `data/private/`. `.gitignore` blocks all `.xlsx`, `.xls`, `.csv` and `.pdf` files except `sample_startup_mis.xlsx`, plus model weights, adapters, datasets and vector stores.
- `sample_startup_mis.xlsx` is a fake company built by `make_sample_mis.py`.

## Dataset design

`generate_raft_data.py` simulates 12-36 months of a startup's P&L, unit economics and pricing plan with a fixed seed. Each example:

- `<financial_context>`: the relevant tables plus 3-6 distractor tables (headcount, cash, cohorts, benchmarks, prior-year P&L, other P&L sections), 3-12 month columns, the same Markdown chunk format the ingester produces.
- A question about one period. Tasks: net revenue, gross margin, CM1/CM2/CM3, EBITDA (from EBIT or net income, adjusted EBITDA, EBITDA margin), break-even (simple, cost components, step-fixed hiring, base/bull/bear), LTV:CAC, CAC payback, runway, burn multiple (ARR bridge), magic number (fiscal quarters), multi-period questions (month-on-month growth, totals and averages over the last 3-6 months, year-to-date from April).
- Founder-style phrasing: extra question templates, casual casing, prefixes ("Quick one:") and suffixes.
- Units: about 20% of companies report their statements in INR lakh, INR crore or USD thousands. Amounts are passed to the calculator as shown and results stay in those units; when scaled amounts meet counts or plain-unit values (S&M in lakh / new customers for CAC), the model converts them first with `from_lakh` / `from_crore` / `from_thousands` / `from_millions`.
- Edge cases: negative gross margin, seasonal churn spike, zero churn, capitalized R&D, deferred revenue, step-fixed costs, price below variable cost, cash-positive months (runway not applicable), shrinking ARR (burn multiple undefined), and missing-data examples (only where the value cannot be derived from other rows).
- Assistant turn 1: `<think>` with each input's table/row/column, the tables ignored, the formulas; then a `finance_calculator` call.
- Tool turn: results from `common/tools.py`.
- Assistant turn 2: short `<think>` check, then a Markdown CFO report that copies the tool's display values.

Validation re-runs every tool call, checks it against ground truth computed separately from the simulated data, checks that every input number appears in the context, checks that every input quoted in `<think>` is the exact cell of that row, and checks that every result appears in the report.

**Prompt contracts.** The tool description lists the calculator's formulas, so adding formulas changes what a model sees. Each trained adapter stores its contract in `contract.json` (adapters without one are `v1`); `run_assistant.py` and `eval_model.py` show each model the tool list it was trained with (`--contract` overrides).

## Layout

```
common/            formulas.py (all finance math), tools.py (calculator), prompts.py (system prompt + tool schema),
                   tables.py (chunk rendering), schemas.py, audit.py (faithfulness check), scoring.py (eval metrics),
                   periods.py ("last month" -> latest period in the data), units.py (lakh / crore / '000 detection)
configs/           data.yaml, train.yaml, rag.yaml
generate_raft_data.py  train_qwen3.py  ingest_financials.py  rag_pipeline.py  make_sample_mis.py  run_assistant.py
eval_model.py  eval_cases.py  export.py
documentation/     memory.md (project memory), Changelog.md
reports/           evaluation reports
tests/
```

## Evaluation

`eval_model.py` scores two suites:
- **Synthetic** (oracle context): fresh examples from a seed never used in training, 20 per task family by default.
- **Real-world**: 31 founder-style questions: 26 on the sample MIS (rw-22..26 are multi-period) and 5 on a lakh-format copy of it (`make_sample_mis.py --lakh`, built into `outputs/` on each run; CAC needs the lakh marketing rows converted). `--v1-compat` uses the original 21 about the sample MIS (`eval_cases.py`), answered through the full retrieval pipeline. Expected values are computed in Python from the workbook's data.

**Units in real files:** the ingester reads scaled units from a `Units:` row, a title row or header cell ("P&L (₹ in lakhs)"), a PDF caption, or a units note such as "All figures in INR lakh unless stated", and writes them into every chunk as `Units: INR lakh`.

**Period guard:** if a question names a month or quarter that is not in the uploaded data (for example Mar-25 when the MIS covers Apr-25 to Mar-26), `run_assistant.py` answers "data not available" without calling the model, so another period's numbers are never reported under the asked period's name. The real-world eval uses the guard by default; `--no-period-guard` measures the raw model.

Metrics: numeric accuracy vs Python ground truth, value recall, formula accuracy, context faithfulness (no numbers absent from context/tool output), report consistency, end-to-end, missing-data handling, edge-case handling, false refusals, latency. Comparisons: fine-tuned vs base (synthetic), RAG + fine-tune vs RAG-only (real-world), adapter vs quantized GGUF (`--backend server`).

## Disclaimer

This tool produces calculations from the data you provide. It is not financial, investment, tax or legal advice. Verify figures before using them in decisions, filings or investor communication. Everything runs locally; no document or question leaves your machine. See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations and component licenses.

## Status

See `documentation/memory.md` (build progress and roadmap) and `documentation/Changelog.md`.
