# af-finance-llm

A startup CFO assistant that reads your MIS / board-pack tables and answers finance questions (net revenue, CM1-CM3, EBITDA, break-even, LTV:CAC, payback) without making up numbers.

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

# 4. End-to-end assistant
python run_assistant.py                                  # demo questions on the sample MIS
python run_assistant.py --question "What is our CAC payback for Feb-26?"
python run_assistant.py --files data/private/our_mis.xlsx --question "..."
python run_assistant.py --retrieval-only                 # no LLM, just show retrieved chunks
```

## Data and privacy

- Put real MIS / board packs in `data/private/`. `.gitignore` blocks all `.xlsx`, `.xls`, `.csv` and `.pdf` files except `sample_startup_mis.xlsx`, plus model weights, adapters, datasets and vector stores.
- `sample_startup_mis.xlsx` is a fake company built by `make_sample_mis.py`.

## Dataset design

`generate_raft_data.py` simulates 12-36 months of a startup's P&L, unit economics and pricing plan with a fixed seed. Each example:

- `<financial_context>`: the relevant tables plus 3-6 distractor tables (headcount, cash, cohorts, benchmarks, prior-year P&L, other P&L sections), 3-12 month columns, the same Markdown chunk format the ingester produces.
- A question about one period. Tasks: net revenue, CM1/CM2/CM3, EBITDA (from EBIT or net income, adjusted EBITDA, EBITDA margin), break-even (simple, cost components, step-fixed hiring, base/bull/bear), LTV:CAC, CAC payback.
- Edge cases: negative gross margin, seasonal churn spike, zero churn, capitalized R&D, deferred revenue, step-fixed costs, price below variable cost, and about 9% missing-data examples.
- Assistant turn 1: `<think>` with each input's table/row/column, the tables ignored, the formulas; then a `finance_calculator` call.
- Tool turn: results from `common/tools.py`.
- Assistant turn 2: short `<think>` check, then a Markdown CFO report that copies the tool's display values.

Validation re-runs every tool call, checks it against ground truth computed separately from the simulated data, checks that every input number appears in the context, and checks that every result appears in the report.

## Layout

```
common/            formulas.py (all finance math), tools.py (calculator), prompts.py (system prompt + tool schema),
                   tables.py (chunk rendering), schemas.py, audit.py (faithfulness check)
configs/           data.yaml, train.yaml, rag.yaml
generate_raft_data.py  train_qwen3.py  ingest_financials.py  rag_pipeline.py  make_sample_mis.py  run_assistant.py
tests/
```

## Not done yet

- Held-out evaluation report in `reports/` (base vs fine-tuned, RAG-only vs RAG + fine-tune).
- Export: merged 16-bit for vLLM and GGUF for Ollama / llama.cpp.
