# Startup Finance LLM: RAG + Unsloth Fine-Tuning

End-to-end startup financial analysis engine. A fine-tuned model (Unsloth QLoRA on Qwen3-4B) supplies CFO-style reasoning, a table-aware hybrid RAG pipeline supplies company-specific numbers, and deterministic Python executes all arithmetic.

## Architecture Rules (do not violate)

| Layer | Responsibility |
|---|---|
| Fine-tuned model | Methodology, formula selection, step-by-step reasoning in `<think>`, strict output format. Never does multi-digit arithmetic in its head. |
| RAG pipeline | Company-specific facts: P&L rows, ledger items, burn, cohort tables, benchmarks. |
| Python tools | All exact arithmetic. The model emits a `finance_calculator` tool call (formula names + variables as JSON); `common/tools.py` computes. |

## Financial Definitions (use everywhere)

- Net Revenue = Gross Revenue - (Discounts + Returns)
- Gross Margin = (Net Revenue - COGS) / Net Revenue
- Contribution Margin = Net Revenue - Variable Costs
  - CM1 = Net Revenue - COGS
  - CM2 = CM1 - Variable Ops Costs (fulfillment, logistics, payment gateway, packaging)
  - CM3 = CM2 - Variable (performance) Marketing
- EBIT = Net Income + Interest + Taxes
- EBITDA = EBIT + Depreciation + Amortization (amortization includes amortization of capitalized R&D)
- Adjusted EBITDA = EBITDA +/- documented one-off items only
- Net Burn = Cash Out - Cash In (monthly); Runway = Cash Balance / Net Burn
- CAC = Total Sales & Marketing Spend / New Customers
- LTV = (ARPU x Gross Margin) / Churn Rate
- LTV:CAC = LTV / CAC
- Payback Period = CAC / (ARPU x Gross Margin) in months
- Break-even units = Fixed Costs / (Price - Variable Cost per unit), rounded up for whole units
- Burn Multiple = Net Burn / Net New ARR
- Magic Number = (Current Qtr Revenue - Prior Qtr Revenue) x 4 / Prior Qtr S&M Spend

## Hardware / Environment

- Dev machine: Windows 11, RTX 5050 Laptop GPU (**8 GB VRAM**), 24 GB RAM.
- Conda env: `unsloth_env` (Python 3.12, torch cu130 build, unsloth, trl).
- Base model: `unsloth/Qwen3-4B-unsloth-bnb-4bit` (fits 8 GB for QLoRA). Larger models (7B+) need a cloud GPU.
- During inference, keep the LLM on GPU and run embedding/reranker models on CPU unless VRAM allows otherwise.

## Coding Conventions

- Python 3.11+, type hints, `pydantic` for schemas, `pytest` for tests.
- Every financial formula lives in `common/formulas.py`. All other code imports from it. No inline reimplementations.
- The tool schema, system prompt and context rendering live in `common/` so data generation, training and inference stay byte-identical.
- **Programmatic ground truth:** Python generates the synthetic startup numbers AND computes every answer. Reasoning traces are template-generated around pre-computed values. Never let an LLM generate numbers and also compute results in free text.
- Fixed random seeds for reproducibility; configs in `configs/*.yaml`.
- Missing variables must be reported explicitly ("churn rate not provided"), never silently assumed.
- Real financial files go in `data/private/` (git-ignored). Only `sample_startup_mis.xlsx` (synthetic) is committed.

## Repository Layout (hybrid: flat entry-point scripts + shared `common/` package)

```text
af-finance-llm/
├── CLAUDE.md / README.md / pyproject.toml / .gitignore
├── configs/                  # data.yaml, train.yaml, rag.yaml
├── common/
│   ├── formulas.py           # single source of truth for all finance math
│   ├── schemas.py            # pydantic models: CalcStep, CalcRequest, TableChunk, ...
│   ├── tools.py              # safe finance_calculator dispatcher (no eval)
│   ├── prompts.py            # system prompt + tool schema shared by data/train/inference
│   └── tables.py             # Markdown table/chunk rendering shared by data gen + ingest
├── generate_raft_data.py     # synthetic RAFT dataset -> dataset.jsonl (+ math validation)
├── train_qwen3.py            # Unsloth QLoRA SFT -> qwen3_4b_finance_lora/
├── ingest_financials.py      # Excel (pandas) / PDF (Docling) -> header-carrying Markdown chunks
├── rag_pipeline.py           # BM25 + dense (bge-large) + RRF + bge-reranker
├── make_sample_mis.py        # builds sample_startup_mis.xlsx
├── run_assistant.py          # ingest -> retrieve -> model -> tool call -> CFO report
└── tests/
```

## Build Status

1. Git + .gitignore - done
2. RAFT data generator (1,000 examples, Qwen3 `<think>` + tool calls, 0 validation mismatches) - done
3. QLoRA training script - done (2 epochs ~3.5 h on the RTX 5050, peak ~5.6 GB VRAM, max_seq_length 6144)
4. Ingestion + hybrid RAG - done (dense model and reranker score metadata summaries, not raw digits)
5. Sample MIS + end-to-end assistant - done

Notes learned the hard way:
- Qwen tokenizes numbers digit by digit, so financial tables are token-heavy. `max_context_chars` in data.yaml keeps examples under 6k tokens.
- On Windows, the xformers wheel did not match torch cu130; it is uninstalled (unsloth falls back to PyTorch attention).

Remaining from the original plan: held-out evaluation report (`reports/`) comparing base vs fine-tuned and RAG-only vs RAG+FT, and export (merged 16-bit for vLLM, GGUF for Ollama/llama.cpp).

## Definition of Done

- All tests pass; dataset validation reports zero mismatches.
- A single CLI command runs: ingest docs -> ask question -> grounded, tool-verified answer.
- Evaluation report (Markdown) generated in `reports/`.
- README documents setup, GPU requirements, and the commands for each phase.
