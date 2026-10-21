# Model Evaluation Report

- Date: 2026-10-21 22:57
- Base model: `unsloth/Qwen3-4B-unsloth-bnb-4bit`; adapter: `qwen3_4b_finance_lora`
- Synthetic: seed 9001, 20 per task family (every 7th is a missing-data case)
- Decoding: temperature 0.0, max_new_tokens 1536
- Real-world period guard: on (absent periods answered without the model)
- Mode: scorer self-test on gold answers

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

| Metric | gold answers |
|---|---|
| Numeric accuracy | 100.0% |
| Value recall | 100.0% |
| Formula accuracy | 100.0% |
| Context faithfulness | 100.0% |
| Report consistency | 100.0% |
| End-to-end correct | 100.0% |
| Missing-data handling | 100.0% |
| Edge-case handling | 100.0% |
| False refusals | 0.0% |
| Cases (answerable / missing) | 108 / 12 |
| Mean latency per case | n/a |

### gold answers: by task

| Task | Cases | Numeric acc. | Formula acc. | Faithful | Missing handled | End-to-end |
|---|---|---|---|---|---|---|
| break_even | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contribution_margin | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ebitda | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ltv_cac | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| net_revenue | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| payback | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
