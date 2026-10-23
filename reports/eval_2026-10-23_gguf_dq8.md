# Model Evaluation Report

- Date: 2026-10-23 18:09
- Base model: `unsloth/Qwen3-4B-unsloth-bnb-4bit`; adapter: `qwen3_4b_finance_lora_v2`
- Synthetic: seed 9001, 10 per task family (every 7th is a missing-data case)
- Decoding: temperature 0.0, max_new_tokens 1536
- Real-world period guard: on (absent periods answered without the model)
- Prompt contract: v2
- Backend: http://localhost:8080/v1

## Metric definitions

- **Numeric accuracy**: non-missing cases where every expected value was computed (and not-computable metrics were flagged).
- **Value recall**: share of expected values found in the calculator results.
- **Formula accuracy**: used the right formula(s).
- **Context faithfulness**: no tool inputs or report numbers that are absent from the context/tool output.
- **Report consistency**: report copies the calculator's display values.
- **End-to-end correct**: numeric + report + faithfulness all pass.
- **Missing-data handling**: missing-data cases: no tool call and names the missing input.
- **Edge-case handling**: zero churn / price below unit cost etc. reported as not computable.
- **False refusals**: answerable cases where no calculator call was made (lower is better).

## Synthetic held-out (oracle context)

| Metric | server: af-finance-qwen3-4b |
|---|---|
| Numeric accuracy | 97.0% |
| Value recall | 97.0% |
| Formula accuracy | 99.0% |
| Context faithfulness | 100.0% |
| Report consistency | 100.0% |
| End-to-end correct | 95.5% |
| Missing-data handling | 81.8% |
| Edge-case handling | 100.0% |
| False refusals | 1.0% |
| Cases (answerable / missing) | 99 / 11 |
| Mean latency per case | 16s |

### server: af-finance-qwen3-4b: by task

| Task | Cases | Numeric acc. | Formula acc. | Faithful | Missing handled | End-to-end |
|---|---|---|---|---|---|---|
| break_even | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| burn_multiple | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contribution_margin | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ebitda | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 90.0% |
| gross_margin | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 90.0% |
| ltv_cac | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| magic_number | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| multi_period | 10 | 77.8% | 100.0% | 100.0% | 100.0% | 80.0% |
| net_revenue | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| payback | 10 | 88.9% | 88.9% | 100.0% | 100.0% | 90.0% |
| runway | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Failures (server: af-finance-qwen3-4b, first 5):

- `raft-1020006` ebitda: called the calculator anyway - "I need the Jan '24 EBITDA number for the board deck. Show the bridge."
- `raft-1050007` payback: no calculator call; wrong/missing values: ['cac', 'net_revenue', 'gross_margin', 'payback_months'] - "What is our CAC payback period for Apr '25?"
- `raft-1090006` gross_margin: called the calculator anyway - "Calculate gross profit and gross margin for Mar 2025."
- `raft-1100004` multi_period: wrong/missing values: ['total'] - "Hey, sum up performance marketing spend over the 4 months to Jan '25."
- `raft-1100008` multi_period: wrong/missing values: ['total'] - "What was total gross revenue for the 4 months ending Sep-22?"

## Real-world questions on sample MIS (hybrid RAG context)

| Metric | server: af-finance-qwen3-4b |
|---|---|
| Numeric accuracy | 96.3% |
| Value recall | 96.3% |
| Formula accuracy | 100.0% |
| Context faithfulness | 100.0% |
| Report consistency | 100.0% |
| End-to-end correct | 96.8% |
| Missing-data handling | 100.0% |
| Edge-case handling | n/a |
| False refusals | 3.7% |
| Cases (answerable / missing) | 27 / 4 |
| Mean latency per case | 15s |

### server: af-finance-qwen3-4b: by task

| Task | Cases | Numeric acc. | Formula acc. | Faithful | Missing handled | End-to-end |
|---|---|---|---|---|---|---|
| break_even | 2 | 100.0% | 100.0% | 100.0% | n/a | 100.0% |
| contribution_margin | 4 | 100.0% | 100.0% | 100.0% | n/a | 100.0% |
| ebitda | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ltv_cac | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| multi_period | 6 | 83.3% | 100.0% | 100.0% | n/a | 83.3% |
| net_revenue | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| payback | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Failures (server: af-finance-qwen3-4b, first 1):

- `rw-25` multi_period: no calculator call; wrong/missing values: ['total'] - "How many new customers did we add in Q3 FY26?"

## Retrieval recall (real-world, top-k chunks)

Mean recall of needed sections: **100.0%**

| Case | Task | Needed | Recall | Retrieved (in order) |
|---|---|---|---|---|
| rw-01 | net_revenue | Revenue | 100.0% | Revenue, Profit Summary, Cost of Revenue, Customer Metrics, Operating Expenses |
| rw-02 | net_revenue | Revenue | 100.0% | Revenue, Variable Costs, FTEs, Marketing, Unit Cost & Fixed Cost Plan |
| rw-03 | contribution_margin | Cost of Revenue, Revenue, Variable Costs | 100.0% | Variable Costs, Revenue, Cost of Revenue, Customer Metrics, Profit Summary |
| rw-04 | contribution_margin | Cost of Revenue, Revenue, Variable Costs | 100.0% | Variable Costs, Cost of Revenue, Revenue, Unit Cost & Fixed Cost Plan, D&A, Interest & Tax |
| rw-05 | contribution_margin | Cost of Revenue, Marketing, Revenue, Variable Costs | 100.0% | Cost of Revenue, Marketing, Revenue, Variable Costs, Unit Cost & Fixed Cost Plan |
| rw-06 | ebitda | D&A, Interest & Tax, Profit Summary | 100.0% | D&A, Interest & Tax, Profit Summary, Revenue, Cost of Revenue, Marketing |
| rw-07 | ebitda | D&A, Interest & Tax, Profit Summary | 100.0% | D&A, Interest & Tax, Profit Summary, Customer Metrics, Unit Cost & Fixed Cost Plan, Bank |
| rw-08 | ebitda | D&A, Interest & Tax, Profit Summary, Revenue | 100.0% | Revenue, D&A, Interest & Tax, Profit Summary, Customer Metrics, Cost of Revenue |
| rw-09 | ltv_cac | Customer Metrics | 100.0% | Customer Metrics, Cost of Revenue, Unit Cost & Fixed Cost Plan, Marketing, Revenue |
| rw-10 | ltv_cac | Customer Metrics | 100.0% | Customer Metrics, Marketing, Revenue, Profit Summary, Cost of Revenue |
| rw-11 | payback | Customer Metrics | 100.0% | Customer Metrics, Marketing, Revenue, Cost of Revenue, Variable Costs |
| rw-12 | payback | Customer Metrics | 100.0% | Customer Metrics, Marketing, Profit Summary, D&A, Interest & Tax, Revenue |
| rw-13 | break_even | Unit Cost & Fixed Cost Plan | 100.0% | Unit Cost & Fixed Cost Plan, Cost of Revenue, Revenue, Variable Costs, Marketing |
| rw-14 | break_even | Unit Cost & Fixed Cost Plan | 100.0% | Unit Cost & Fixed Cost Plan, Variable Costs, Cost of Revenue, Operating Expenses, Profit Summary |
| rw-15 | net_revenue | Revenue | 100.0% | Revenue, Profit Summary, Cost of Revenue, Customer Metrics, D&A, Interest & Tax |
| rw-20 | net_revenue | Revenue | 100.0% | Revenue, Profit Summary, Cost of Revenue, Customer Metrics, Operating Expenses |
| rw-21 | contribution_margin | Cost of Revenue, Revenue, Variable Costs | 100.0% | Variable Costs, Cost of Revenue, Unit Cost & Fixed Cost Plan, Revenue, Customer Metrics |
| rw-22 | multi_period | Revenue | 100.0% | Revenue, Customer Metrics, FTEs, Cost of Revenue, Profit Summary |
| rw-23 | multi_period | Operating Expenses | 100.0% | Revenue, Customer Metrics, Operating Expenses, Cost of Revenue, Profit Summary |
| rw-24 | multi_period | Revenue | 100.0% | Unit Cost & Fixed Cost Plan, Customer Metrics, Revenue, D&A, Interest & Tax, Profit Summary |
| rw-25 | multi_period | Customer Metrics | 100.0% | Customer Metrics, Unit Cost & Fixed Cost Plan, Variable Costs, Cost of Revenue, Revenue |
| rw-26 | multi_period | Customer Metrics | 100.0% | Customer Metrics, Unit Cost & Fixed Cost Plan, Cost of Revenue, FTEs, Variable Costs |
| rw-16 | ltv_cac | Customer Metrics | 100.0% | Customer Metrics, Marketing, Cost of Revenue, Revenue, Bank |
| rw-17 | net_revenue | Revenue | 100.0% | Revenue, Profit Summary, Customer Metrics, Unit Cost & Fixed Cost Plan, Cost of Revenue |
| rw-18 | payback | Customer Metrics | 100.0% | Customer Metrics, Marketing, Revenue, Unit Cost & Fixed Cost Plan, Profit Summary |
| rw-19 | ebitda | D&A, Interest & Tax, Profit Summary | 100.0% | D&A, Interest & Tax, Profit Summary, Customer Metrics, Bank, Revenue |
| rl-01 | net_revenue | Revenue | 100.0% | Revenue, Profit Summary, Customer Metrics, Cost of Revenue, Operating Expenses |
| rl-02 | ebitda | D&A, Interest & Tax, Profit Summary | 100.0% | D&A, Interest & Tax, Profit Summary, Customer Metrics, Revenue, Cost of Revenue |
| rl-03 | ltv_cac | Customer Metrics, Marketing | 100.0% | Customer Metrics, Cost of Revenue, Marketing, Unit Cost & Fixed Cost Plan, Revenue |
| rl-04 | payback | Customer Metrics, Marketing | 100.0% | Customer Metrics, Marketing, Profit Summary, D&A, Interest & Tax, Revenue |
| rl-05 | multi_period | Operating Expenses | 100.0% | Customer Metrics, Revenue, Operating Expenses, Cost of Revenue, Profit Summary |
