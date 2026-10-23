# Model Card: af-finance-qwen3-4b (draft)

> Draft. v2 adapter evaluated 2026-10-23 (current default); v1 numbers kept for comparison.

## Summary

A LoRA adapter for Qwen3-4B that acts as a startup CFO assistant. Given tables from a company's MIS or board pack (retrieved by the project's RAG pipeline), it identifies the period and line items, chooses the right formula, and calls a deterministic Python calculator (`finance_calculator`). It writes a short CFO report that copies the calculator's results. It is designed never to do arithmetic itself and to say clearly when an input is missing.

| | |
|---|---|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` (Qwen3-4B, 4-bit) |
| Method | QLoRA (Unsloth), r=32, alpha=32, all linear layers, loss on assistant turns only |
| Training data | v2 (default): 2,698 synthetic training examples, 11 task families, 1 epoch. v1: 900 examples, 6 task families. Template-generated with Python ground truth |
| Context length used | up to 6,144 tokens in training; 8,192 at inference |
| Output format | Qwen3 `<think>` reasoning + tool call, then a Markdown CFO report |
| Developed by | Aeryal Flow |

## Intended use

- Founders, finance teams and analysts asking questions about their own MIS / P&L / unit-economics tables.
- Supported metrics (v1 adapter): net revenue, CM1/CM2/CM3, EBITDA (incl. adjusted EBITDA and EBITDA margin), break-even (incl. step-fixed costs and scenarios), CAC, LTV, LTV:CAC, CAC payback.
- Added in the v2 adapter: runway, burn multiple, magic number, gross margin, multi-period questions (month-on-month and quarter-on-quarter growth, totals and averages over the last 3-6 months, year-to-date), statements in INR lakh / INR crore / USD thousands (with explicit unit conversion when lakh amounts meet counts, e.g. CAC), more varied founder phrasing, and questions about months that are not in the data.
- Must be used with the calculator tool and retrieved context. Without the tool, numeric answers are not reliable.

## Out of scope

- Investment, tax, legal or accounting advice.
- Forecasts or valuations. Multi-period questions with the v1 adapter (the v2 adapter covers them).
- Documents other than tabular financial reports; scanned images without OCR quality checks.

## Training data

Synthetic companies (12-36 months of P&L, unit economics, pricing, cash and ARR data) generated with a fixed seed. Each example contains the relevant tables plus 3-6 distractor tables, one question, a reasoning trace, a calculator call, the calculator's result, and a report. Edge cases: negative gross margin, seasonal churn spikes, zero churn, capitalized R&D, deferred revenue, step-fixed costs, price below unit cost, and ~9% missing-data examples. Every number is validated against an independent Python calculation. No real company data was used.

## Evaluation

### v2 adapter (current default)

`reports/eval_2026-10-23_v2.md` (greedy decoding; fine-tuned and base model scored on identical inputs; ground truth from Python; prompt contract v2):

| Suite | Metric | Fine-tuned (v2) | Base Qwen3-4B |
|---|---|---|---|
| Synthetic held-out (110 cases, 11 task families, oracle context) | End-to-end correct | **95.5%** | 9.1% |
| | Numeric accuracy | 97.0% | 7.1% |
| | Formula accuracy | 98.0% | 10.1% |
| | Context faithfulness (no invented numbers) | 100.0% | 43.4% |
| | Missing-data handling | 81.8% | 63.6% |
| | Edge-case handling | 100.0% | 80.0% |
| | False refusals | 1.0% | 63.6% |
| Real-world (31 questions: 26 on the sample MIS, 5 on an INR-lakh copy; full RAG pipeline with the period guard) | End-to-end correct | **96.8%** (30/31) | 12.9% |
| | Numeric accuracy (answerable) | 96.3% | 3.7% |
| | Context faithfulness | 100.0% | 25.9% |
| | False refusals | 3.7% | 51.9% |

Latency on an RTX 5050 laptop GPU (Hugging Face generation, 4-bit): ~60 s per question for the fine-tuned model.

Compared with v1:
- On v1's 6 task families the v2 adapter scores 57/60 (95%) vs v1's 54/60 (90%), on the same seed with the v2 question generator.
- On v1's 21 real-world questions v2 scores 21/21 (v1: 16/21 raw; 20/21 counting the period guard). The off-by-one column read (rw-12) is gone.
- All 5 INR-lakh questions pass, including the two that need a lakh-to-rupee conversion before computing CAC.
- The 4 questions about months that are not in the data are answered by the deterministic period guard for both models (the base model's 4 real-world passes are exactly these).

What the v2 failures show (5 synthetic, 1 real-world):
- Missing-data cases where the model still called the calculator (adjusted EBITDA, gross margin), one false refusal (payback), one adjusted-EBITDA margin computed on the wrong revenue line, one 4-month range total.
- "How many new customers did we add in Q3 FY26?": the pipeline spelled the quarter out as Oct-25, Nov-25, Dec-25 and the model quoted all three values correctly, then treated the question as year-to-date growth and refused for lack of last year's data. No training question names a quarter as an explicit list of months; planned for the next data version.

### Quantized GGUF (llama.cpp)

Same 141 cases, served by llama-server (b11734, CUDA, RTX 5050) and called through the OpenAI-compatible API:

| Export | Synthetic end-to-end | Real-world end-to-end | Faithfulness (syn / real) | Latency per case |
|---|---|---|---|---|
| LoRA adapter, Hugging Face 4-bit (reference) | 95.5% | 96.8% | 100% / 100% | ~60 s |
| **q8_0 GGUF, LoRA merged into the dequantized 4-bit base (`export.py`, shipped)** | **95.5%** | **96.8%** | **100% / 100%** | **~16 s** |
| q4_k_m GGUF, Unsloth merge into the original bf16 weights | 81.8% | 87.1% | 92.9% / 92.6% | ~14 s |

Reports: `reports/eval_2026-10-23_gguf_dq8.md`, `reports/eval_2026-10-23_gguf.md`.

- The shipped q8_0 file reproduces the adapter's scores exactly (same per-suite numbers; the only case-level differences are one synthetic case each way) at about 4x the speed. It needs ~5.4 GB of GPU memory with flash attention.
- The adapter was trained (QLoRA) on top of the 4-bit-quantized base. Merging it into the original 16-bit weights, as Unsloth's export does, adds the 4-bit rounding error back (about 9% mean relative difference on the quantized layers) and costs ~14 points: the model starts doing arithmetic itself (margin percentages, leakage) instead of in the calculator. A q8_0 file from that merge showed the same failures, so quantization was not the cause.
- A 4-bit GGUF is not available yet: re-quantizing the merged weights (already on the 4-bit nf4 grid) to q4_k_m rounds them a second time and was clearly worse in a 5-case check. q5_k_m / q6_k have not been evaluated.

### v1 adapter

`reports/eval_2026-10-22_v1.md` (greedy decoding; fine-tuned and base model scored on identical inputs; ground truth from Python):

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

Quantized GGUF vs adapter: see "Quantized GGUF (llama.cpp)" above.

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
