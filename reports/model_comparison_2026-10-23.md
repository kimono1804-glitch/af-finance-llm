# Model comparison: af-finance-qwen3-4b vs finance-capable models (2026-10-23)

Two kinds of numbers, kept apart on purpose:

- **A. Our benchmark** (141 cases: 110 synthetic + 31 questions on a sample MIS, same prompt, calculator tool and scorer for every model). Only models run through this harness are comparable here.
- **B. Published finance benchmarks** (other teams, other tasks, other scoring). Useful to see where frontier and finance-tuned models stand, **not** comparable with A or with each other. No model was downloaded or run for part B.

## A. Our benchmark (measured)

| Model | Size / where it runs | Synthetic end-to-end | Real-world end-to-end | No invented numbers (syn / real) | False refusals (syn) | Time per question |
|---|---|---|---|---|---|---|
| **af-finance-qwen3-4b v2, q8_0 GGUF** (shipped) | 4B, local RTX 5050 (llama.cpp) | **95.5%** | **96.8%** (30/31) | **100% / 100%** | 1.0% | ~16 s |
| af-finance-qwen3-4b v2, LoRA adapter (HF 4-bit) | 4B, local | 95.5% | 96.8% | 100% / 100% | 1.0% | ~60 s |
| af-finance-qwen3-4b v1 (6 task families) | 4B, local | 90.0% (60 cases) | 76.2% raw (21 cases) | 100% / 100% | 3.7% | ~147 s |
| Base Qwen3-4B (same prompt + tool, no fine-tune) | 4B, local | 9.1% | 12.9% (4/31, all from the period guard) | 43.4% / 25.9% | 63.6% | ~112 s |
| Frontier models (Claude Opus 5.5 / Sonnet 5.5, GPT, Gemini) | API | not run | not run | - | - | - |

Reports: `eval_2026-10-23_gguf_dq8.md`, `eval_2026-10-23_v2.md`, `eval_2026-10-22_v1.md`.
"End-to-end" = right formula, every value matches Python ground truth, report copies the calculator output, and no number in the answer is absent from the source tables.

Frontier models can be run through the same harness with `eval_model.py --backend claude --claude-model claude-opus-5-5 --contract v2 --config configs/eval_v2.yaml --tag cmp` (needs `ANTHROPIC_API_KEY`; estimated a few dollars to ~$15 per model for 141 cases) or `--backend server --server-url <OpenAI-compatible API> --server-api-key-env <VAR>`.

## B. Published finance benchmarks (online sources, not re-run)

### B1. Vals AI Finance Agent v2.1 - agentic analyst work over SEC filings (449 test tasks, tools incl. EDGAR search and a calculator, LLM-jury partial-credit grading; updated 2026-10-22)

| Model | Score |
|---|---|
| Claude Opus 5.5 | 67.85% |
| Gemini 4 Argon | 67.66% |
| Wonder Full | 67.58% |
| Claude Sonnet 5.5 | 67.28% |
| Gemini 3.8 Flash | 65.76% |
| DeepSeek V4.1 Flash | 65.43% |
| Claude Haiku 5.5 | 65.37% |
| GLM 5.3 | 65.00% |
| GPT-6.1 Sol | 61.82% |
| Mistral Large 4 | 61.38% |
| MiniMax-M3 | 54.34% |
| Inkling Small | 44.78% |

Hardest categories top out at ~35% (financial modelling) and ~36% (precedents).

### B2. FinanceReasoning (hard subset) - numeric financial calculations from tables / time series (238 questions, pass if within 0.2% of the reference; AIMultiple, 2026-10)

| Model | Accuracy | Cost for 238 questions |
|---|---|---|
| Claude Opus 5.5 | 91.60% | $3.25 |
| Gemini 4 Argon | 90.76% | $2.65 |
| GPT-5.6 Sol | 90.34% | $3.85 |
| Claude Opus 5 | 89.92% | $6.08 |
| Mistral Large 4 | 87.39% | $3.31 |
| DeepSeek V4.1 Flash | 86.97% | $0.26 |
| Claude Haiku 5.5 | 85.71% | $0.15 |
| GPT-OSS-120B (open) | 81.09% | $0.06 |
| Llama 4 Maverick (open) | 75.21% | $0.10 |
| DeepSeek R1 0528 (open) | 62.18% | - |
| Mistral Small 3.2 24B (open) | 41.18% | $0.04 |

### B3. Finance-tuned small models - FinQA / ConvFinQA (Fin-R1 paper, LLM-judged, up to 1,000 sampled questions)

| Model | Size | FinQA | ConvFinQA | 5-benchmark average |
|---|---|---|---|---|
| Fin-R1 (finance RL-tuned Qwen2.5) | 7B | 76.0 | 85.0 | 75.2 |
| DeepSeek-R1 | 671B | 71.0 | 82.0 | 78.2 |
| Qwen2.5-32B-Instruct | 32B | 72.0 | 78.0 | 73.8 |
| Qwen2.5-14B-Instruct | 14B | 68.0 | 77.0 | 71.4 |
| Qwen2.5-7B-Instruct | 7B | 60.0 | 66.0 | 65.6 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 55.0 | 62.0 | 58.0 |

Other reference points: GPT-4o FinQA 72.5 (Fino1 leaderboard); Fino1-8B (Llama 3.1 8B, finance-tuned) FinQA 60.9; FinanceBench (open-book 10-K Q&A) frontier models reported in the mid-to-high 80s by aggregator sites whose own figures disagree (treat as unverified).

## Reading the two tables together

- **Same-test evidence:** on its own task, the fine-tuned 4B model goes from 9% (base) to ~96%, with no invented numbers. That gain comes from the fine-tune plus the deterministic calculator, retrieval and period guard.
- **No same-test evidence vs frontier models yet.** Frontier models score 85-92% on public numeric finance maths (B2) and ~65-68% on agentic filing research (B1), but those are broader and harder tasks than ours. A frontier model would very likely also score high on our narrow MIS tasks; how high is unknown until it is run through part A.
- **Scope:** our model covers 11 metric families (net revenue, CM1-3, EBITDA, break-even, LTV:CAC, payback, runway, burn multiple, magic number, gross margin, multi-period) from MIS / board-pack tables. It does not do open-ended filing research, valuation, forecasting or general finance Q&A, which is what B1-B3 measure. It would score poorly on FinQA-style questions outside its formulas.
- **Where it is strong regardless of the scores:** runs fully on a laptop GPU (data never leaves the machine), ~16 s per answer, no per-question cost, every number comes from Python and is checked against the source tables, and it refuses when an input or period is missing.

## Sources

- Vals AI Finance Agent v2.1 leaderboard: https://www.vals.ai/benchmarks/fabv2-1 (also v2: https://www.vals.ai/benchmarks/fabv2)
- AIMultiple finance LLM benchmark (FinanceReasoning hard subset): https://aimultiple.com/finance-llm
- Fin-R1 paper (arXiv 2503.16252, v5): https://arxiv.org/html/2503.16252v5 ; repo https://github.com/sufe-aiflm-lab/fin-r1
- Fino1 / Open FinLLM reasoning leaderboard: https://thefinai-open-finllm-reasoning-leaderboard.hf.space
- FinanceBench aggregators (unverified): https://financebenchmark.ai/benchmarks/financebench , https://github.com/meetdewey/financebench-eval
