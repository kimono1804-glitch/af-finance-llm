# Changelog: af-finance-llm

Every change to code, data, config, environment or models, newest first. Times are IST (UTC+05:30).
Format: `## YYYY-MM-DD HH:MM IST - title`, then what changed, which files, and why.
Entries before 14:09 on 2026-10-21 were reconstructed from file timestamps, git history and the session transcript.

---

## 2026-10-21 23:08 IST - Session paused for the day
- Renamed the `--gold` scorer self-test report to `reports/eval_2026-10-21_gold-selftest.md` so it is not mistaken for a model result.
- User to commit and push. Base-model eval still running in the background (pid 32708, ~75 cases left, ~2.5 h); resume checklist in memory.md section 1.

## 2026-10-21 23:00 IST - Slowdown root cause found: E-core scheduling; CPU boost helper
- Base-model eval slowed again at ~22:48 (4.5 then 3.0 tok/s; one turn hit the 240 s limit on raft-1000004) with no other load from this session. GPU 9% busy; Python one core at 100%; per-core counters: P-cores (0-15) mostly idle at ~56% max frequency, E-cores (16-23) busy (core 19 at 94%).
- 22:55: pinned the running eval (pid 32708) to the P-cores with `ProcessorAffinity = 0xFFFF`: GPU 9% -> 42%, next turn 14.7 tok/s (base model's normal speed). Root cause: Windows scheduling the GPU-feeding thread on efficiency cores. The earlier git-scan and priority theories were wrong.
- New `common/winperf.py`: `boost_process()` turns off power throttling (EcoQoS) for the process and pins it to the fastest core class (detected with GetLogicalProcessorInformationEx; here mask 0xFFFF = 16 threads). Called at the start of `main()` in `run_assistant.py`, `eval_model.py`, `train_qwen3.py`, `export.py`. `AF_NO_CPU_BOOST=1` turns it off. No admin rights needed.
- raft-1000004 (base) result is invalid (time limit); rerun after the base run by deleting its row.
- Checks: `eval_model.py --gold` 100% on every metric (includes the new period-gap missing cases); fast tests pass.

## 2026-10-21 22:55 IST - Fixes from the eval: period guard, row decimals, period-gap training examples
Code was edited while the base-model eval was running; that process had already loaded its modules, so the v1 run is unaffected (no guard).
- **Period guard (no retraining).** `common/periods.py`: `asked_periods` (strict month/quarter matcher: `Mar-25`, `March 2025`, `Sep '25`, `Q4 of FY2026`; ignores "margin 40", "decrease 10", "FY27 plan", lowercase "may 25"), `quarter_months` (Indian FY), `missing_periods` (a quarter counts as present if its label or all 3 months are in the data), `coverage`, `period_not_available_report`. `run_assistant.py`: `period_guard(question, known_periods, sources)` returns a no-model trace ("Data not available: Mar-25 ... data covers Apr-25 to Mar-26"); `main()` calls it after relative-period resolution, before retrieval.
- `eval_model.py`: real-world cases carry the guard trace; `run_model` uses it instead of calling the model; `--no-period-guard` measures the raw model; report header states the guard mode. Checked on all 21 real-world questions: fires on rw-16..19 only; never on the 17 answerable ones or the 4 demo questions.
- **Row decimals.** `common/tables.py` `align_row_decimals`, applied in `render_chunk` (ingester AND generator): if any number in a row has decimals, whole numbers in that row get the same decimals (`1,240` -> `1,240.00`). Labels and percentages untouched. Changes the training-data rendering (v2 data only; PROMPT_VERSION unchanged).
- **Period-gap training examples.** `generate_raft_data.py`: `absent_months(c)` (up to 12 months before / 6 after the window), `period_gap_example(...)`: question asks for an absent month; one assistant turn: think lists the covered periods, says the month is absent and that another period's column must not be used; report "data missing ... tables cover X to Y"; meta `missing_kind: "period"` (cell-type missing examples get `missing_kind: "cell"`). Applies to month-period tasks incl. runway and burn multiple (which had no missing examples). `validate_example` checks the asked month is absent from the context and named in the answer. `configs/data.yaml`: `missing_period_share: 0.4`.
- Trial: 1,500 examples, 0 mismatches; 140 missing (84 cell, 56 period-gap).
- Tests: `tests/test_tables.py` (new), period tests (phrasings, lookalikes, Indian-FY quarters, missing periods, report), dataset period-gap test, `period_guard` test. 58 fast tests pass.
- README: period guard + `--no-period-guard`.
- Build progress: 74% -> 75% (milestone 8: 4 -> 5).

## 2026-10-21 22:40 IST - Fine-tuned evaluation complete (70/81); base model running
- All 81 fine-tuned cases ran at normal speed after the 22:15 resume (~11-12 tok/s; no time-limit hits).
- Result: 70/81 end-to-end. Synthetic 54/60; real-world 16/21 (normal 16/17, missing-period 0/4).
- New failures: rw-16..rw-19 (asked period absent from the MIS): the model used another period's column and labelled it as the asked period (rw-16: Mar-25 asked, Mar-26 values used). rw-12: ARPU taken from the neighbouring Aug-25 column; that Sep-25 cell renders as `1,240` vs 2-decimal neighbours. Scorer checked: both failures are real, not tolerance artifacts.
- Base-model run started (81 cases). No code changes.

## 2026-10-21 22:16 IST - Git scans ruled out; evaluation resumed again
- The `C:\Users\nevin` repo was not in the Source Control view. The looping `git status --untracked-files=all` comes from VS Code's Copilot agent host (`agentHostMain.js` `computeSessionFileDiffs`, process `copilot-runtime.exe` parent), not the Git extension.
- Speed benchmark at 22:13 (scratchpad `gen_bench.py`: v1 adapter, 2,860-token prompt, 4 x 200 new tokens greedy): 10.5-11.4 tok/s WHILE the git scans continued. So the scans did not cause the 21:37-22:02 slowdown; the earlier entry's "likely cause" was wrong. Cause still unknown (intermittent).
- The workspace `.vscode/settings.json` git settings stay (harmless; hide the home repo from Source Control).
- 22:15: resumed `eval_model.py --models both --n-per-task 10 --tag v1` from 67/81 fine-tuned rows; watching per-turn speed and `(TIME LIMIT)` hits.

## 2026-10-21 21:35 IST - Evaluation resumed
- Restarted `eval_model.py --models both --n-per-task 10 --tag v1` (log appended to `outputs/eval_v1_log.txt` after a `RESUMED` marker). 65 cached fine-tuned rows are reused; 16 fine-tuned + 81 base cases remain (~1.5-2 h at ~1 min/case).
- Generation ran at 2.5-3.2 tok/s (normal ~12). Not VRAM spill (130 MB shared), not battery/GPU throttling (AC, no throttle flags, GPU util 5%). py-spy: main thread in Unsloth's Triton nf4 GEMV launch (CPU-side). Raising process priority to High did not help.
- Likely cause: VS Code's Git extension runs `git status --untracked-files=all` on the accidental home-dir repo `C:\Users\nevin` every few seconds; each scan walks 535,421 untracked files and takes ~62 s (the project repo takes 0.13 s).
- Workspace `.vscode/settings.json` (outside the repo): added `"git.ignoredRepositories": ["C:\\Users\\nevin"]` and `"git.openRepositoryInParentFolders": "never"`. Takes effect after closing that repository in Source Control or reloading the window.
- 22:02: stopped the eval (pid 8152) because slow turns reached 238.6 s, right at the 240 s `max_time_per_turn` cut-off; truncated turns would be scored as model failures. 67/81 fine-tuned rows cached (rw-06, rw-07 OK). Resume with the same command once generation is back to ~12 tok/s.

## 2026-10-21 21:10 IST - Evaluation paused by user
- Stopped the eval process (pid 7576) on the user's "pause". GPU free. Nothing lost: 65 fine-tuned rows cached in `outputs/eval_runs/fine_tuned_v1.jsonl`; base model not started.
- Interim (fine-tuned): synthetic 54/60 end-to-end (90%), real-world 5/5. Failure types: false missing-data refusals (2), unclear missing-input refusal (1), S&M taken from the quarterly distractor in LTV:CAC (3).
- Resume: `python eval_model.py --models both --n-per-task 10 --tag v1`.

## 2026-10-21 19:59 IST - Evaluation restarted cleanly (19:55)
- Clarification: the first run was not frozen. Case 1 (raft-1000000) finished OK at 19:48:56 after 4,877 s; the process was stopped at ~19:50 during case 2. That row is cached and reused.
- Restarted run at 19:55: normal speed (11.4-11.9 tok/s, 57 s/case). Per-turn speed now logged.

## 2026-10-21 19:55 IST - Evaluation stalled -> guard added, restarted
- First eval run (started 18:25) produced only 2 generation turns of case 1 in ~80 min (one `generate()` at ~18% GPU, 1 CPU core; no VRAM spill; process alive, `py-spy dump` showed it inside Unsloth's fast inference). Stopped it (pid 39832; results file was empty). Log kept as `outputs/eval_v1_log_stalled.txt`.
- Reproduction outside the eval: the same case ran in 55 s (~11 tok/s, greedy and sampled alike). Root cause not found.
- `run_assistant.py` `FinanceLLM`: `max_time` per `generate()` call (transformers `max_time`), `log_rounds` per-turn speed logging, `last_stats`.
- `eval_model.py`: `run_model(..., max_time)`; local models get `max_time_per_turn` and per-turn logging.
- `configs/eval.yaml`: `max_time_per_turn: 240`.
- `pip install py-spy` in unsloth_env (diagnostics).
- Correction: the "Sample Startup" header comes from the source file name, not pure invention; generator fix stays.
- Restarted: `eval_model.py --models both --n-per-task 10 --tag v1`. Tests: 50 passed.

## 2026-10-21 18:26 IST - Demo with trained adapter; company-name fix; evaluation started
- `run_assistant.py` demo with the v1 adapter (`outputs/demo_run.txt`): 4/4 correct and faithful (CM1/CM2, LTV:CAC, EBITDA, break-even with hires). 64-92 s per question.
- Found: reports invent a company name ("Sample Startup") because v1 training reports carried the simulated company name, which never appears in the context.
- `generate_raft_data.py`: report header is now `Source: <file> | Currency: <cur>` (no company name). Applies to v2 data; the v1 adapter keeps the flaw until retrained.
- Started evaluation: `eval_model.py --models both --n-per-task 10 --tag v1` -> `reports/eval_2026-10-21_v1.md` (log `outputs/eval_v1_log.txt`).
- Build progress: 72% -> 74% (milestone 5: 7 -> 9).

## 2026-10-21 18:18 IST - v1 adapter trained
- Resumed run completed at 18:17 IST: 226/226 steps (2 epochs), 98.8 min for the resumed part, peak VRAM 5.90 GB, final train loss ~0.009.
- Saved `qwen3_4b_finance_lora/` (adapter_model.safetensors 264 MB, tokenizer, chat_template.jinja, loss_history.json, loss_curve.png). Exit traceback is the known harmless multiprocess resource_tracker message.
- Started `run_assistant.py` demo (4 questions on the sample MIS) -> `outputs/demo_run.txt`.
- Build progress: 67% -> 72% (milestone 3: 10 -> 15).

## 2026-10-21 16:55 IST - Varied period phrasing in questions; quarter normalisation; regex bug fixed
- `generate_raft_data.py`: `MONTH_NAMES` + `say_period()`; `build_example` rewrites the period in the question (Mar-26 / March 2026 / Mar 2026 / Mar '26; Q4 FY26 / Q4 of FY26 / Q4 FY2026). Tables and reasoning keep the table label. Removed the ambiguous "April 25" style after spotting it in samples.
- `rag_pipeline.py`: `normalize_periods` also maps `FY2026` -> `FY26` and `Q4 of FY26` -> `Q4 FY26` (`_FY_RE`, `_Q_OF_RE`).
- Bug: the heredoc patch wrote literal backspace bytes instead of `` in those regexes; fixed with an exact edit; repo scanned, no other control characters.
- `tests/test_ingest_rag.py`: normalisation asserts for "Mar '26", "March 2026", "Q4 of FY2026".
- Checks: 50 tests pass; 1,500 v2 trial examples 0 mismatches; real-world retrieval recall 100%.
- Training (resumed) at step 120/226, ~46 s/step, past the old stall point.

## 2026-10-21 16:46 IST - Model card, disclaimer, CI
- Added `MODEL_CARD.md` (draft; evaluation section to fill from reports/; component licenses; project license not chosen yet).
- `README.md`: Disclaimer section (not financial advice, local-only, link to model card).
- Added `.github/workflows/tests.yml` (fast tests + 300-example dataset validation on ubuntu / Python 3.12 with 6 light packages). Verified in a clean venv (scratchpad, Python 3.14.8): 50 passed, 300 examples 0 mismatches.
- Resumed training confirmed healthy: step 104 -> 105 in 46 s, loss 0.013.
- Build progress: 65% -> 67% (milestone 9: 2 -> 4).

## 2026-10-21 16:37 IST - Training stalled (VRAM spill) -> resumed from checkpoint-100
- Run 1 slowed from ~50 s/step to ~1,600 s/step after the epoch-1 eval (steps 113 -> 117 in ~1 h). Cause: the training process spilled 1.78 GB into Windows "shared GPU memory" (system RAM).
- Stopped the training process (pid 9808, started by Claude at 13:30). Lost steps 101-117 (~15 min of work).
- `train_qwen3.py`: added `--resume` (`trainer.train(resume_from_checkpoint=True)`) and `--no-eval`.
- Relaunched: `train_qwen3.py --resume --no-eval` from `outputs/checkpoint-100`; log `outputs/train_log.txt`; old log renamed `outputs/train_log_run1_stalled.txt`.
- memory.md: VRAM-spill gotcha + NVIDIA "Sysmem Fallback Policy" mitigation; progress unchanged at 65%.

## 2026-10-21 15:08 IST - Training: epoch 1 eval
- v1 run reached step 113 (epoch 1). eval_loss 0.0128 on the 100-example test split (train loss ~0.013): no overfitting gap. Recorded in memory.md section 6.

## 2026-10-21 14:59 IST - Documentation refresh after build session
- `documentation/memory.md` rewritten: progress 45% -> 65%, new files, results, decisions, environment, next steps.
- Build progress: 65% (milestones: 10 + 10 + 10 + 10 + 7 + 8 + 4 + 4 + 2).

## 2026-10-21 14:58 IST - Changelog header repaired
- The 14:15 entry had been inserted inside the "Format:" line (insertion searched for the first "## "). Moved it to its own entry below. Future inserts must anchor on the first line starting with "## 20".

## 2026-10-21 14:55 IST - README updated
- `README.md`: 10 task families, PDF board-pack builder, "last month" questions, evaluation commands and metrics, export + server-mode commands, slow tests, new files in layout, status pointer to documentation/.

## 2026-10-21 14:52 IST - PDF ingestion verified + slow test
- Built a board-pack PDF from the sample data and ingested it with Docling (first run downloaded `docling-project/docling-models` and `docling-layout-heron`). 9 chunks; P&L sections identical to Excel (incl. negative EBIT), percent rows intact, commentary as a text chunk.
- Added `tests/test_pdf_ingest.py` (`@pytest.mark.slow`, ~2.5 min). Default suite 50 passed; slow suite 1 passed.

## 2026-10-21 14:50 IST - Relative periods ("last month") + 2 real-world cases
- Added `common/periods.py` (`resolve_relative_periods`: latest/previous month and quarter -> periods present in the data) and `tests/test_periods.py`.
- `run_assistant.py`: rewrites the question before retrieval and prints a NOTE.
- `eval_model.py`: real-world suite applies the same resolution.
- `eval_cases.py`: added rw-20 ("net revenue last month" -> Mar-26) and rw-21 ("CM2 in the previous month" -> Feb-26). Retrieval recall 100% on 21 cases.

## 2026-10-21 14:45 IST - Four new task families in the generator (v2 data, not trained yet)
- `generate_raft_data.py`: simulator now produces cash flow (opening/in/out/closing with equity raises), ARR bridge (open/new/expansion/churned/close), `net_revenue` series and `ym`; new tables `cash_flow_table`, `arr_table`, `quarterly_table` (Indian-FY quarters via `fy_quarters`); new plans `plan_runway`, `plan_burn_multiple` (bridge / open_close), `plan_magic_number`, `plan_gross_margin`; new labels; the fake "Cash Position" distractor replaced by the consistent cash-flow table; relevant tables removed from the distractor pool; ARR and quarterly tables added as distractors.
- Runway and burn multiple get no missing-data examples (inputs derivable from other rows).
- `configs/data.yaml`: task weights runway 1.0, burn_multiple 1.0, magic_number 1.0, gross_margin 0.75; edges cash_positive 0.08, arr_decline 0.08. Running the generator now yields a 10-task (v2) dataset; `dataset.jsonl` on disk is still v1.
- `configs/eval.yaml`: `tasks` list pinned to the 6 v1 tasks; `eval_model.py` reads it.
- `tests/test_dataset.py`: task check reads config; new test for the 4 families with forced edge cases.
- Trial: 1,500 v2 examples, 0 validation mismatches.

## 2026-10-21 14:40 IST - Packages added
- `unsloth_env`: `pip install fpdf2 matplotlib` (fpdf2 2.8.9, matplotlib 3.11.2, contourpy, cycler, fonttools, kiwisolver, pyparsing). numpy/torch unchanged. matplotlib lets `train_qwen3.py` save `loss_curve.png` at the end of the current run.

## 2026-10-21 14:38 IST - Board-pack PDF builder
- `make_sample_mis.py`: moved P&L sections to module constant `PL_SECTIONS`; added `build_board_pack_pdf()` and `--pdf PATH` (fpdf2; commentary page, 12-month P&L, unit economics).

## 2026-10-21 14:36 IST - Server backend (llama-server / Ollama / vLLM)
- `run_assistant.py`: `ServerLLM` (OpenAI-compatible chat API with tools; message conversion with tool_call ids; reply -> Qwen3 text); flags `--backend local|server --server-url --server-model`.
- `eval_model.py`: `--backend server` evaluates an exported model; `run_model` now takes an LLM factory.
- Added `tests/test_server_backend.py` (fake server round trip).
- `.gitignore`: added `exports/`.

## 2026-10-21 14:33 IST - Export script
- Added `export.py`: merged 16-bit (`--merged`) and GGUF (`--gguf q4_k_m ...`) via Unsloth; writes `exports/gguf/Modelfile` and `exports/SERVING.md`. Found Unsloth's prebuilt CUDA13 llama.cpp (b11734) with `llama-server.exe` and `llama-quantize.exe` in `~/.unsloth/llama.cpp/build/bin/Release/`. No Ollama/cmake on this machine.

## 2026-10-21 14:30 IST - Evaluation system
- Added `common/scoring.py` (Gold, score, aggregate, metrics), `eval_cases.py` (19 real-world questions at first), `eval_model.py` (synthetic + real-world suites, ft vs base, resumable runs, Markdown report), `configs/eval.yaml`, `tests/test_scoring.py`.
- `make_sample_mis.py`: pricing values moved to `PRICING` constant (shared with eval_cases).
- `run_assistant.py`: refactored into `run_conversation()`; `FinanceLLM(None, ...)` loads the base model on purpose; `.label`.
- Scorer fixes found by the self-test: numbers quoted in calculator error messages are allowed; a call where every step errors still counts as a call.
- Scorer self-test: 100% on 360 gold cases. Retrieval recall 98.2% -> 100% after adding the "ebitda margin" query expansion in `rag_pipeline.py`.
- Tests: 44 -> 45 passed.

## 2026-10-21 14:15 IST - Data sizing guidance recorded
- `documentation/memory.md`: added section 11 (test-set size vs training-data size, margins of error, recommended eval set ~700 + 30-50 hand-written real-world questions, training data diversity over volume). No code or data changed.
- Build progress: 45% (unchanged). Training at step ~60/226.

## 2026-10-21 14:09 IST - Documentation system and session hooks
- Added `documentation/memory.md` (full project memory: progress %, architecture, environment, file map, dataset contract, results, decisions, gaps, preferences, commands).
- Added `documentation/Changelog.md` (this file).
- Added `.claude/settings.json` with two hooks:
  - SessionStart -> `.claude/hooks/session_start.sh`: injects memory.md and the newest 200 changelog lines into every new session.
  - Stop -> `.claude/hooks/stop_docs_check.sh`: blocks finishing a turn (once) if any project file is newer than the docs, listing the changed files.
- Added workspace-level `C:\Users\nevin\Desktop\aeryal flow\finance llm\.claude\settings.json` (same hooks, pointing into the repo) and `...\finance llm\CLAUDE.md` (pointer), because VS Code opens the parent folder.
- Added `.gitattributes` (`*.sh text eol=lf`); converted hook scripts to LF (CRLF breaks bash).
- `CLAUDE.md`: added "Session start" and "Documentation rules" sections and the build-progress requirement.
- Why: user asked for persistent memory + changelog that auto-update every prompt and replace codebase analysis at session start, plus a build % after each session.
- Hooks pipe-tested with Windows-style $CLAUDE_PROJECT_DIR: SessionStart prints 304 lines; Stop blocks when docs are stale and stays quiet when `stop_hook_active` is true. `jq` is not installed (use the env python for JSON).
- Build progress: 45%.

## 2026-10-21 14:02 IST - User commit `b8fbfdf` "testrun+modelsetup"
- The user committed all work from steps 2-5 (24 files, +3,382 lines): common/, configs/, tests/, generate_raft_data.py, train_qwen3.py, ingest_financials.py, rag_pipeline.py, make_sample_mis.py, run_assistant.py, sample_startup_mis.xlsx, README.md, CLAUDE.md.

## 2026-10-21 13:45 IST - CLAUDE.md build status
- `CLAUDE.md`: steps 2-5 marked done; notes on digit tokenization and the xformers removal.

## 2026-10-21 13:43 IST - Retrieval fix: rerank and embed digit-free chunk summaries
- `rag_pipeline.py`: added `chunk_summary()` (title, sheet, section, row labels, periods, units). Dense embeddings and the cross-encoder now score summaries instead of raw tables; dense and reranker queries get the finance expansion as "(needs: ...)". BM25 still uses full text. Store creation moved inside `load_models`.
- Why: for "CM1 and CM2 for Mar-26" the reranker dropped the Variable Costs table (scores ~0.00-0.03) because 12 columns of digits drown semantic models.
- Result: every demo question now gets its required tables in the top 3. Tests: 39 passed.

## 2026-10-21 ~13:34 IST - Full training run started
- `train_qwen3.py` full run (2 epochs, 226 steps) in background; log at `outputs/train_log.txt`.
- Deleted the 5-step smoke-test adapter (`qwen3_4b_finance_lora/`) and `outputs/checkpoint-5` so nothing loads a half-trained model.
- Masking check: 327/1,826 tokens of example 0 carry loss, starting at `<think>`.
- Loss: 1.121 (step 5), 0.711 (10), 0.298 (20), 0.128 (30), 0.064 (40). ~45-49 s/step, ETA ~16:30 IST.

## 2026-10-21 ~13:20 IST - Training smoke test passed
- `train_qwen3.py --max-steps 5`: train_loss 1.12, eval_loss 0.8986 on 100 test examples, peak VRAM 5.64 GB, 7.8 min including first-run compile and eval.

## 2026-10-21 ~13:00 IST - Models downloaded
- HF cache now has `unsloth/Qwen3-4B-unsloth-bnb-4bit` (3.4 GB), `BAAI/bge-large-en-v1.5` (1.3 GB), `BAAI/bge-reranker-large` (~2.2 GB).

## 2026-10-21 12:52 IST - Training script tweaks + environment cleanup
- `train_qwen3.py`: `train_on_responses_only(..., num_proc=1)` (Windows-safe); prints how many tokens of example 0 carry loss and the first 40 trained tokens.
- Environment: `pip uninstall xformers` (0.0.35 built for torch 2.10+cu128; did not load with torch 2.14.1+cu130).

## 2026-10-21 ~12:50 IST - Environment install finished
- `unsloth_env`: torch 2.14.1+cu130 and torchvision 0.29.1+cu130 (replaced CPU builds; CUDA verified on RTX 5050), unsloth 2026.10.7, unsloth_zoo, trl 1.15.0, peft 0.21.2, bitsandbytes 0.50.2, triton-windows 3.8.0.post29, datasets 4.8.5, chromadb 1.5.9, pytest 9.1.1, cut_cross_entropy, torchao, etc. transformers changed 5.20.2 -> 5.17.0 (unsloth pin).
- Tests: 39 passed.

## 2026-10-21 12:50 IST - Context budget lowered, dataset regenerated
- `configs/data.yaml`: `max_context_chars` 8000 -> 5500 (8000 never triggered; contexts were all < 7.3k chars).
- `dataset.jsonl` regenerated: tokens median 3,613, max 5,821, 0 over 6,144. 0 validation mismatches.

## 2026-10-21 ~12:45 IST - Token budget fixes
- Measured with the Qwen3 tokenizer: median 3,613, max 6,859 tokens; 291 examples > 4,096 (Qwen tokenizes digits one by one).
- `generate_raft_data.py`: drop extra distractors (never below min 3) while context exceeds `max_context_chars`.
- `configs/data.yaml`: added `max_context_chars: 8000`.
- `configs/train.yaml`: `max_seq_length` 4096 -> 6144.
- Verified the rendered Qwen3 template: `<think>` precedes `<tool_call>`, tool results render as user `<tool_response>` turns (masked), final turn includes `<think>`.

## 2026-10-21 12:42 IST - README
- `README.md` rewritten: architecture, setup (CUDA torch first), commands for every step, data/privacy, dataset design, layout, not-done list.

## 2026-10-21 12:41 IST - Generator aligned with real MIS
- `generate_raft_data.py`: month windows 3-6 -> 3-12 columns; unit-economics and pricing table titles now match the ingester ("Unit Economics - Customer Metrics", "Pricing & Cost Structure - Unit Cost & Fixed Cost Plan"); validator now uses `common/audit.py`.

## 2026-10-21 12:40 IST - Tests for dataset, ingestion, retrieval
- Added `tests/test_dataset.py`, `tests/test_ingest_rag.py`.

## 2026-10-21 12:39 IST - Assistant CLI and faithfulness audit
- Added `run_assistant.py` (end-to-end loop with tool calls and audit printout).
- Added `common/audit.py` (`unsupported_inputs` etc.).

## 2026-10-21 12:38 IST - Sample MIS workbook
- Added `make_sample_mis.py`; generated `sample_startup_mis.xlsx` (5 sheets, 11 chunks after ingestion).

## 2026-10-21 12:36 IST - Ingestion + hybrid RAG
- Added `ingest_financials.py`, `rag_pipeline.py`, `configs/rag.yaml`.

## 2026-10-21 12:35 IST - First full dataset
- `dataset.jsonl`: 1,000 examples, validated, deterministic (two runs, identical MD5 e83c4b40...).

## 2026-10-21 12:34 IST - Calculator precision fix
- `common/tools.py`: keep full precision for chained `@refs`; round to 6 dp only in JSON output. Was causing a 4e-6 drift (payback example raft-00027).
- `tests/test_tools.py`: use `pytest.approx`.

## 2026-10-21 ~12:30 IST - RAFT generator
- Added `generate_raft_data.py` and `configs/data.yaml` (6 task families, edge cases, missing-data examples, distractors, validation).
- `common/formulas.py` + `common/tools.py`: added `percent_of`.

## 2026-10-21 12:24-12:28 IST - Shared package and tests
- Added `common/__init__.py`, `formulas.py`, `schemas.py`, `tools.py`, `tables.py`, `prompts.py`; `tests/__init__.py`, `tests/test_formulas.py`, `tests/test_tools.py`.

## 2026-10-21 12:23 IST - Scaffold commit `85ada95`
- Added `.gitignore` (verified: sample xlsx tracked; real xlsx/pdf and dataset.jsonl ignored), `CLAUDE.md` (updated for Qwen3-4B, 8 GB GPU, hybrid layout), `pyproject.toml`.

## 2026-10-21 12:19 IST - Environment audit and install started
- Found: repo already cloned (`c390c9e`), `unsloth_env` had CPU-only torch and no unsloth/trl, RTX 5050 8 GB with CUDA 13.3 driver.
- Started background install of CUDA torch + unsloth + trl + deps.

## 2026-10-21 11:53 IST - Initial commit `c390c9e`
- GitHub repo created with README only.
