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
python train_qwen3.py --max-steps 5     # smoke test
python train_qwen3.py                   # -> qwen3_4b_finance_lora/ (+ loss_curve.png)

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

# 6. Export for serving (exports/, git-ignored)
python export.py --gguf q4_k_m                           # GGUF for llama.cpp / Ollama (+ Modelfile, SERVING.md)
python export.py --gguf q4_k_m --merged                  # also merged 16-bit for vLLM
# then serve it and point the assistant (or the evaluator) at the server:
llama-server -m exports/gguf/<file>.gguf --jinja -ngl 99 -c 8192 --port 8080 --alias af-finance-qwen3-4b
python run_assistant.py --backend server --server-url http://localhost:8080/v1
python eval_model.py --backend server --tag gguf         # measure quantization loss
```

Slow tests (Docling PDF ingestion): `pytest -m slow`.

## Data and privacy

- Put real MIS / board packs in `data/private/`. `.gitignore` blocks all `.xlsx`, `.xls`, `.csv` and `.pdf` files except `sample_startup_mis.xlsx`, plus model weights, adapters, datasets and vector stores.
- `sample_startup_mis.xlsx` is a fake company built by `make_sample_mis.py`.

## Dataset design

`generate_raft_data.py` simulates 12-36 months of a startup's P&L, unit economics and pricing plan with a fixed seed. Each example:

- `<financial_context>`: the relevant tables plus 3-6 distractor tables (headcount, cash, cohorts, benchmarks, prior-year P&L, other P&L sections), 3-12 month columns, the same Markdown chunk format the ingester produces.
- A question about one period. Tasks: net revenue, gross margin, CM1/CM2/CM3, EBITDA (from EBIT or net income, adjusted EBITDA, EBITDA margin), break-even (simple, cost components, step-fixed hiring, base/bull/bear), LTV:CAC, CAC payback, runway, burn multiple (ARR bridge), magic number (fiscal quarters).
- Edge cases: negative gross margin, seasonal churn spike, zero churn, capitalized R&D, deferred revenue, step-fixed costs, price below variable cost, cash-positive months (runway not applicable), shrinking ARR (burn multiple undefined), and missing-data examples (only where the value cannot be derived from other rows).
- Assistant turn 1: `<think>` with each input's table/row/column, the tables ignored, the formulas; then a `finance_calculator` call.
- Tool turn: results from `common/tools.py`.
- Assistant turn 2: short `<think>` check, then a Markdown CFO report that copies the tool's display values.

Validation re-runs every tool call, checks it against ground truth computed separately from the simulated data, checks that every input number appears in the context, and checks that every result appears in the report.

## Layout

```
common/            formulas.py (all finance math), tools.py (calculator), prompts.py (system prompt + tool schema),
                   tables.py (chunk rendering), schemas.py, audit.py (faithfulness check), scoring.py (eval metrics),
                   periods.py ("last month" -> latest period in the data)
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
- **Real-world**: 21 founder-style questions about the sample MIS (`eval_cases.py`), answered through the full retrieval pipeline. Expected values are computed in Python from the workbook's data.

**Period guard:** if a question names a month or quarter that is not in the uploaded data (for example Mar-25 when the MIS covers Apr-25 to Mar-26), `run_assistant.py` answers "data not available" without calling the model, so another period's numbers are never reported under the asked period's name. The real-world eval uses the guard by default; `--no-period-guard` measures the raw model.

Metrics: numeric accuracy vs Python ground truth, value recall, formula accuracy, context faithfulness (no numbers absent from context/tool output), report consistency, end-to-end, missing-data handling, edge-case handling, false refusals, latency. Comparisons: fine-tuned vs base (synthetic), RAG + fine-tune vs RAG-only (real-world), adapter vs quantized GGUF (`--backend server`).

## Disclaimer

This tool produces calculations from the data you provide. It is not financial, investment, tax or legal advice. Verify figures before using them in decisions, filings or investor communication. Everything runs locally; no document or question leaves your machine. See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations and component licenses.

## Status

See `documentation/memory.md` (build progress and roadmap) and `documentation/Changelog.md`.
