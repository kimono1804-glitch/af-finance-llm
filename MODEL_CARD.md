# Model Card: af-finance-qwen3-4b (draft)

> Draft. Evaluation numbers will be filled in from `reports/eval_*.md` after the evaluation run.

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

To be filled from `reports/`:
- Synthetic held-out (fresh seed, oracle context): numeric accuracy, formula accuracy, faithfulness, missing-data and edge-case handling; fine-tuned vs base.
- Real-world questions on the sample MIS through the full RAG pipeline: same metrics, RAG + fine-tune vs RAG only.
- Quantized GGUF vs adapter.
- Retrieval recall on the real-world questions: 100% (31/31: 26 on the sample MIS, 5 on a lakh-format copy) as of 2026-10-22.

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
