# Model Card: af-finance-qwen3-4b (draft)

> Draft. v1 evaluation numbers below; v2 numbers will be added after the v2 run.

## Summary

A LoRA adapter for Qwen3-4B that acts as a startup CFO assistant. Given tables from a company's MIS or board pack (retrieved by the project's RAG pipeline), it identifies the period and line items, chooses the right formula, and calls a deterministic Python calculator (`finance_calculator`). It writes a short CFO report that copies the calculator's results. It is designed never to do arithmetic itself and to say clearly when an input is missing.

| | |
|---|---|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` (Qwen3-4B, 4-bit) |
| Method | QLoRA (Unsloth), r=32, alpha=32, all linear layers, loss on assistant turns only |
| Training data | v1: 900 synthetic training examples (6 task families). v2 (prepared): 3,000 examples, 11 task families. Template-generated with Python ground truth |
| Context length used | up to 6,144 tokens in training; 8,192 at inference |
| Output format | Qwen3 `<think>` reasoning + tool call, then a Markdown CFO report |
| Developed by | Aeryal Flow |

## Intended use

- Founders, finance teams and analysts asking questions about their own MIS / P&L / unit-economics tables.
- Supported metrics (v1 adapter): net revenue, CM1/CM2/CM3, EBITDA (incl. adjusted EBITDA and EBITDA margin), break-even (incl. step-fixed costs and scenarios), CAC, LTV, LTV:CAC, CAC payback.
- Added in the v2 training data: runway, burn multiple, magic number, gross margin, multi-period questions (month-on-month and quarter-on-quarter growth, totals and averages over the last 3-6 months, year-to-date), statements in INR lakh / INR crore / USD thousands (with explicit unit conversion when lakh amounts meet counts, e.g. CAC), more varied founder phrasing, and questions about months that are not in the data.
- Must be used with the calculator tool and retrieved context. Without the tool, numeric answers are not reliable.

## Out of scope

- Investment, tax, legal or accounting advice.
- Forecasts or valuations. Multi-period questions with the v1 adapter (v2 data covers them).
- Documents other than tabular financial reports; scanned images without OCR quality checks.

## Training data

Synthetic companies (12-36 months of P&L, unit economics, pricing, cash and ARR data) generated with a fixed seed. Each example contains the relevant tables plus 3-6 distractor tables, one question, a reasoning trace, a calculator call, the calculator's result, and a report. Edge cases: negative gross margin, seasonal churn spikes, zero churn, capitalized R&D, deferred revenue, step-fixed costs, price below unit cost, and ~9% missing-data examples. Every number is validated against an independent Python calculation. No real company data was used.

## Evaluation

v1 adapter, `reports/eval_2026-10-22_v1.md` (greedy decoding; fine-tuned and base model scored on identical inputs; ground truth from Python):

| Suite | Metric | Fine-tuned (v1) | Base Qwen3-4B |
|---|---|---|---|
| Synthetic held-out (60 cases, 6 task families, oracle context) | End-to-end correct | **90.0%** | 15.0% |
| | Numeric accuracy | 90.7% | 9.3% |
| | Formula accuracy | 96.3% | 9.3% |
| | Context faithfulness (no invented numbers) | 100.0% | 40.7% |
| | Missing-data handling | 83.3% | 66.7% |
| | False refusals | 3.7% | 59.3% |
| Real-world (21 questions on the sample MIS, full RAG pipeline, raw model) | End-to-end correct | **76.2%** | 9.5% |
| | Numeric accuracy (answerable) | 94.1% | 0.0% |
| | Context faithfulness | 100.0% | 47.1% |

What the v1 failures show:
- Questions about a month that is not in the file (4 real-world cases): the raw v1 model used another month's column. The assistant now answers these with a deterministic period guard before the model is called (it fires on exactly those 4 questions and none of the others), so the shipped pipeline would score 20/21 on this set; that guarded run has not been measured separately. The v2 data also teaches the model to refuse such questions itself.
- LTV:CAC (3 synthetic cases): S&M taken from a quarterly summary table instead of the monthly rows. v2 data adds an explicit reasoning step for this.
- One off-by-one column read (rw-12) and two false "data missing" refusals.

Not yet measured: the v2 adapter (11 task families, lakh / crore units, multi-period; 110 synthetic + 31 real-world cases), and quantized GGUF vs adapter.

Retrieval recall on the real-world questions: 100% (31/31: 26 on the sample MIS, 5 on a lakh-format copy) as of 2026-10-22.

## Limitations and risks

- Trained only on synthetic data; real MIS files can use layouts, labels and units the model has not seen. Check the printed inputs and the faithfulness check on every answer.
- "Last month" is interpreted as the latest month present in the uploaded data, not the calendar month.
- Units: amounts are passed to the calculator as shown and results stay in the table's units (e.g. INR lakh). The v2 data teaches converting lakh / crore / thousands amounts with `from_lakh` etc. before combining them with counts or plain-unit values; the v1 adapter never saw scaled units. The ingester detects units from a `Units:` row, titles, headers or a units note ("All figures in INR lakh").
- The prompt contract (tool list) must match the adapter: v1 adapters use contract v1, v2 adapters store theirs in `contract.json`.
- The model can still pick a wrong row or period. The assistant prints every input with its table, row and column so a person can verify it.

## Disclaimer

This tool produces calculations from the data you provide. It is not financial, investment, tax or legal advice. Verify figures before using them in decisions, filings or investor communication.

## Privacy

Everything runs locally: documents, embeddings and the model stay on your machine. Nothing is sent to external services.

## Licenses of components

| Component | License |
|---|---|
| Qwen3-4B (base model) | Apache 2.0 |
| Unsloth | Apache 2.0 |
| BAAI/bge-large-en-v1.5, BAAI/bge-reranker-large | MIT |
| Docling | MIT |
| llama.cpp | MIT |
| This project's code and adapter | Not chosen yet (owner to decide) |
