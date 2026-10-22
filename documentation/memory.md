# Project Memory: af-finance-llm

> Single source of truth for Claude Code sessions. Read this file and `documentation/Changelog.md` at the start of every session INSTEAD of analysing the codebase. Open a source file only when a task needs its exact code.
> Update this file at the end of every prompt that changes or teaches anything (enforced by `.claude/hooks/stop_docs_check.sh`).
> Last updated: 2026-10-22 21:02 IST

---

## 1. Build progress toward a releasable v1

**Overall: 85%** (weighted milestones; update after every build session and show it to the user)

| # | Milestone | Weight | Done | Status |
|---|---|---|---|---|
| 1 | Foundations: formulas, schemas, calculator tool, prompts, tests | 10 | 10 | Done. 98 fast tests + 1 slow test pass (2026-10-22 10:45). Prompt contracts v1/v2 (section 7). |
| 2 | Synthetic RAFT dataset (validated) | 10 | 10 | v1 (1,000 ex, 6 tasks) -> v1 adapter. v2 `dataset_v2.jsonl` (3,000 ex, 11 tasks, phrasing variety, quarterly note, lakh/crore, multi-period incl. QoQ; 0 mismatches) -> v2 adapter. |
| 3 | Training pipeline + first trained adapter | 15 | 15 | v1 adapter (2026-10-21, 6 tasks). v2 adapter `qwen3_4b_finance_lora_v2/` (2026-10-22 17:19; 2,698 examples, 11 tasks, 1 epoch, 338 steps, final loss ~0.010, peak VRAM 6.19 GB; contract.json v2). DONE. |
| 4 | Table-aware ingestion + hybrid retrieval | 10 | 10 | Excel and PDF (Docling) verified; PDF chunks identical to Excel. Retrieval recall 100% on 21 real-world questions (rechecked 10:40 after the units change). Units detection for lakh/crore/'000/mn in Excel and PDF. |
| 5 | End-to-end assistant CLI | 10 | 9 | 4/4 demo answers correct with the v1 adapter. Open: 64-92 s per question with local HF generation (GGUF + llama-server should be faster); server backend only tested against a fake server. |
| 6 | Evaluation (`eval_model.py`, report in `reports/`) | 15 | 11 | v1 report `reports/eval_2026-10-22_v1.md`: synthetic 90.0% (base 15.0%), real-world 76.2% raw / 20 of 21 with the period guard (base 9.5%). v2 eval: fine-tuned 137/141 cached (4 to redo), base 5/141 - paused, see Next steps. Scorer fixes (optional intermediates, month counts) + `--rescore`. Open: finish v2 eval, GGUF vs adapter.
| 7 | Export: merged 16-bit + GGUF + Modelfile + serve commands | 10 | 4 | `export.py` + SERVING.md template + `--backend server` done; not run yet. |
| 8 | Hardening: real/messy files, PDFs, more tasks, relative periods, units | 10 | 10 | PDF verified; 4 new task families (not trained yet); "last month"; period guard; row decimals; period-gap examples; founder phrasing variety; lakh/crore/thousands units end to end (calculator, generator, ingestion; not trained yet); multi-period family (MoM growth, n-month totals/averages, YTD; not trained yet) + 5 real-world cases. Lakh-format sample workbook + 5 real-world lakh cases (rl-01..05). Messy Excel layouts (stacked tables, numbers as text incl. Indian grouping / (negatives), months down the rows, currency in titles). QoQ growth from quarterly tables. Open: an actual founder MIS file, messy PDFs. |
| 9 | Packaging & release | 10 | 4 | README, `MODEL_CARD.md` draft, CI verified. Open: project license (owner), install simplification, versioning, HF upload, optional UI/API. |

How to recompute: sum the "Done" column (out of 100). Move points only when the work is verified by tests or run output.

### Next steps (in order)
**Session paused 2026-10-22 21:02 IST by the user; continue tomorrow.** Nothing is running (eval, chain scripts, keep-awake
helpers all stopped). Nothing committed by Claude today.

State on disk:
- v2 adapter `qwen3_4b_finance_lora_v2/` trained and saved (contract v2). v1 adapter `qwen3_4b_finance_lora/` untouched.
- v2 eval cache (`outputs/eval_runs/`): `fine_tuned_v2.jsonl` = 137 rows (all 141 cases except rw-20, rw-21, rw-25, rw-26,
  removed on purpose because they ran with the old period preprocessing; backup with all 141 rows:
  `fine_tuned_v2.before_period_fix.jsonl`). `base_v2.jsonl` = 5 rows (raft-1000000..04). No time-limit hits so far.
- Old-scorer fine-tuned result on all 141 (before the scorer and period fixes): 128/141; most misses were scorer false
  negatives (burn multiple closing-opening route, "9 months" in a YTD report) that the current scorer accepts.

**Resume checklist (tomorrow):**
1. Start the remaining v2 eval (4 fine-tuned real-world cases ~5 min, then 136 base cases ~4.5 h). GPU must be free; keep the
   laptop lid open (the script blocks idle sleep itself):
   `cd "/c/Users/nevin/Desktop/aeryal flow/finance llm/af-finance-llm"`
   `/c/Users/nevin/miniconda3/envs/unsloth_env/python.exe -u eval_model.py --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2 --models both >> outputs/eval_v2_log.txt 2>&1`
   (run in the background; cached cases are skipped; progress lines `[fine-tuned] k/141` / `[base] k/141` in the log;
   watchdog `bash outputs/watch_eval_v2.sh` reports every 10 cases). If short on time: `--models ft` first, base later.
2. When it finishes: delete any row whose turn shows `(TIME LIMIT)` in the log and rerun step 1's command; then write the final
   report with the current scorer: same command plus `--rescore` (no GPU) -> `reports/eval_<date>_v2.md`.
3. Compare v2 with v1 (`reports/eval_2026-10-22_v1.md`; note v2 has 11 task families and 31 real-world cases), add the v2
   numbers to MODEL_CARD.md, update milestone 6.
4. If v2 wins: set `adapter_dir: qwen3_4b_finance_lora_v2` in `configs/train.yaml` (default for run_assistant / export / eval).
5. Export + quantization check: `python export.py --adapter qwen3_4b_finance_lora_v2 --gguf q4_k_m` -> serve with llama-server
   (command in section 10 / `exports/SERVING.md`) -> `eval_model.py --backend server --contract v2 --config configs/eval_v2.yaml --tag gguf`.
6. Then milestone 9 (packaging): license choice (owner), install simplification, version tag, optional HF upload.

Gotchas for tomorrow: start GPU jobs fresh (boost_process: P-cores, no throttling, no sleep); never put backslashes in a
Bash heredoc (write scripts with the Write tool); a chain script must hold its own keep-awake (use `/proc/$$/winpid` for the
Windows pid in Git Bash).

Earlier history: v1 eval DONE 2026-10-22 11:51 (`reports/eval_2026-10-22_v1.md`, synthetic 90.0% vs base 15.0%, real-world 76.2%
raw vs 9.5%). v2 data 3,000 ex / 11 tasks; v2 training 11:51-17:19 (paused 66 min by standby). Optional v1 extras never run:
`eval_model.py --suite realworld --tag v1-guard`; v1 GGUF export.
- `reports/eval_2026-10-21_gold-selftest.md` is the scorer self-test (not a model result).

---

## 2. What the project is

A startup CFO assistant. It reads a founder's MIS / board-pack tables (Excel or PDF) and answers finance questions with tool-verified numbers. Owner: Aeryal Flow (git user `nevinroby`, founders@aeryalflow.com). Solo developer.

Three layers (CLAUDE.md "Architecture Rules", never violate):

| Layer | Job | Where |
|---|---|---|
| Fine-tuned Qwen3-4B (QLoRA via Unsloth) | Reads context, picks period + line items, picks the formula, reasons in `<think>`, emits a `finance_calculator` tool call, writes the CFO report. Never does multi-digit arithmetic. | `train_qwen3.py`, adapter in `qwen3_4b_finance_lora/` |
| Table-aware hybrid RAG | Finds the right table chunks. Every chunk keeps headers, row labels, periods, units, source. | `ingest_financials.py`, `rag_pipeline.py` |
| Deterministic Python | All arithmetic, via `common/formulas.py` only. | `common/tools.py` |

Inference flow: question -> relative-period resolution ("last month" -> latest month in data) -> hybrid retrieval (BM25 + bge-large dense + RRF + bge-reranker; dense/rerank on digit-free chunk summaries) -> top-5 chunks -> Qwen3 `<think>` + tool call -> Python computes -> Qwen3 writes report copying tool display values -> faithfulness audit. Missing input -> model names it, no tool call, no guessing.

---

## 3. Environment and machine

- OS: Windows 11 Home (10.0.26200). Shells: Git Bash (Bash tool) and PowerShell 5.1. `python` and `jq` are NOT on the Git Bash PATH (use the env python for JSON). No Ollama, no cmake installed.
- Hardware: i7-14650HX, 24 GB RAM, **RTX 5050 Laptop GPU 8 GB** (Blackwell). NVIDIA driver 610.47, CUDA UMD 13.3.
- Python env: conda `unsloth_env` at `C:\Users\nevin\miniconda3\envs\unsloth_env\python.exe` (Python 3.12.0). Use this exe explicitly. Base miniconda python `C:\Users\nevin\miniconda3\python.exe` has no project deps (fine for text-editing scripts).
- Key versions in unsloth_env: torch 2.14.1+cu130 (CUDA works), torchvision 0.29.1+cu130, unsloth 2026.10.7, unsloth_zoo 2026.10.7, trl 1.15.0, transformers 5.17.0, peft 0.21.2, bitsandbytes 0.50.2, triton-windows 3.8.0.post29, datasets 4.8.5, accelerate 1.15.0, sentence-transformers 6.1.0, chromadb 1.5.9, qdrant-client 1.19.1, rank-bm25 0.2.2, docling 2.141.0, rapidocr, pandas 3.0.6, openpyxl 3.1.5, pydantic 2.14.1, pytest 9.1.1, pyyaml, **fpdf2 2.8.9, matplotlib 3.11.2** (added 14:40 for the PDF builder and the loss curve).
- xformers was UNINSTALLED (wheel built for torch 2.10+cu128). Unsloth uses PyTorch attention. Optional: `unsloth install-kernels xformers`.
- Harmless noise: `multiprocess resource_tracker ... '_thread.RLock' object has no attribute '_recursion_count'` at exit; HF "symlinks on Windows" warning; "unauthenticated requests to the HF Hub".
- HF cache `C:\Users\nevin\.cache\huggingface\hub\`: `unsloth/Qwen3-4B-unsloth-bnb-4bit` (3.4 GB), `BAAI/bge-large-en-v1.5` (1.3 GB), `BAAI/bge-reranker-large` (~2.2 GB), `docling-project/docling-models` + `docling-layout-heron` (Docling PDF models).
- llama.cpp: Unsloth prebuilt CUDA13 build at `C:\Users\nevin\.unsloth\llama.cpp` (tag b11734); binaries in `build\bin\Release\` (`llama-server.exe`, `llama-quantize.exe`); `convert_hf_to_gguf.py` present. Used by `save_pretrained_gguf` and for serving.
- **Patch-script gotcha (hit again twice on 2026-10-22):** even a quoted heredoc (`<<'EOF'`) through the Bash tool turns `\\b` / `\b` into a backspace byte. Never put backslashes in a heredoc: write scripts and doc entries with the Write tool, run them by path, then scan for control bytes. Fixer: replace byte 0x08 with `\b`. Python patch scripts passed through a Bash heredoc mangle backslash escapes (`\b` became a literal backspace byte 0x08 in rag_pipeline.py). Edit regex/backslash code with the Edit tool, never via heredoc string replacement. Check with `grep -rlP "[\x00-\x08]" --include=*.py .`.
- **Generation slowdown - ROOT CAUSE FOUND (22:56 IST):** Windows runs the eval's single GPU-feeding Python thread on an E-core (i7-14650HX: logical CPUs 0-15 = P-cores, 16-23 = E-cores), probably because the process is background work started from VS Code. Symptoms: 2.5-4.5 tok/s, GPU 5-10% busy, one core 100% (an E-core), py-spy in Unsloth's Triton nf4 GEMV launch. Pinning the running eval to P-cores (affinity 0xFFFF) restored 14.7 tok/s on the next turn. Fix: `common/winperf.py` `boost_process()` (power throttling off via SetProcessInformation + pin to the fastest core class via GetLogicalProcessorInformationEx; no admin; `AF_NO_CPU_BOOST=1` disables) is called at the start of `main()` in run_assistant / eval_model / train_qwen3 / export. Manual fix for an already-running process (PowerShell): `(Get-Process -Id <pid>).ProcessorAffinity = [IntPtr]0xFFFF`. Earlier history: 4,877 s case at 18:25-19:48; slow episode 21:37-22:02; 22:48-22:55. `max_time_per_turn` guard + per-turn speed logging stay.
- Slowdown episode 21:37-22:02 IST: same E-core cause (see above); priority High did not help, affinity did.
- 2026-10-22: the manual affinity pin alone did NOT hold - after 22:55 many base turns ran at ~6 tok/s again until the run stopped (~00:58). With `boost_process()` from process start (affinity + power throttling off) the 10:22 run holds 15.9-16.1 tok/s. So power throttling (EcoQoS) is part of the cause; always start GPU jobs fresh so the boost applies.
- **Laptop sleep (Modern Standby) pauses GPU jobs** (v2 training lost 66 min on 2026-10-22). `boost_process()` now blocks idle sleep while a project script runs (SetThreadExecutionState); for an already-running process use `outputs/keep_awake.ps1 -ProcessId <pid>`. Check sleep with the System event log (Kernel-Power 506/507). Lid close can still sleep.
- Bash tool gotcha: a stray `cat > file` with no heredoc waits on stdin forever. Pass `< /dev/null` to long python commands; write multi-line patch scripts with the Write tool (heredoc quoting broke one patch on 2026-10-22).
- Background noise (not the slowdown cause): VS Code's Copilot agent host (`agentHostMain.js`, `computeSessionFileDiffs`) runs `git status --porcelain=v1 -z --untracked-files=all` back-to-back on the home-dir repo `C:\Users\nevin` (535k untracked files, ~5-10 s warm, one core). Workspace `.vscode/settings.json` got `git.ignoredRepositories: ["C:\\Users\\nevin"]` + `git.openRepositoryInParentFolders: never` (hides it from Source Control; does not stop the agent host).
- Local HF generation speed with the v1 adapter: ~11 tok/s (greedy or sampled), prompt ~2.4k tokens; a typical case = 2 turns, ~640 new tokens, ~55 s.
- Internet ~2 MB/s: big downloads take tens of minutes; run them in the background.
- While training runs, run other Python with `CUDA_VISIBLE_DEVICES=""` so it stays off the GPU. CPU-heavy work slows training a little (49 -> 58 s/step).
- **VRAM spill gotcha (Windows WDDM):** when a process needs more than the 8 GB card, the NVIDIA driver silently spills into system RAM ("shared GPU memory") instead of raising OOM, and training slows ~30x. Seen after the mid-training eval (training pid showed 1.78 GB shared). Check with PowerShell `Get-Counter '\GPU Process Memory(*)\Shared Usage'` or Task Manager. Mitigations: `--no-eval`, smaller max_seq_length, or set NVIDIA Control Panel > Manage 3D settings > "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback" (fails fast with OOM instead).

### Paths and git
- VS Code workspace root: `C:\Users\nevin\Desktop\aeryal flow\finance llm\` (has `.vscode/settings.json` = conda env manager, `.claude/settings.json` hooks pointing into the repo, and a pointer `CLAUDE.md`).
- Repo: `C:\Users\nevin\Desktop\aeryal flow\finance llm\af-finance-llm\` (Git Bash: `/c/Users/nevin/Desktop/aeryal flow/finance llm/af-finance-llm`).
- Remote `https://github.com/kimono1804-glitch/af-finance-llm.git`, branch `main`. Nothing pushed by Claude.
- Commits: `c390c9e` Initial commit -> `85ada95` scaffold (Claude) -> `b8fbfdf` "testrun+modelsetup" (user, 14:02). Everything after 14:02 is uncommitted (docs, hooks, eval, export, periods, new tasks, PDF builder, tests).
- `C:\Users\nevin` (home dir) is an accidental git repo with everything untracked. Never touch it.

---

## 4. File map (purpose of every file)

### Root scripts (entry points)
| File | Purpose |
|---|---|
| `generate_raft_data.py` | (2026-10-22 additions: `vary_question`/`EXTRA_QUESTIONS`/`Q_PREFIXES`/`Q_SUFFIXES`; `scale_company`/`pick_money_unit`/`money`/`NON_MONEY_KEYS`/`UNIT_LABELS`/`FROM_UNIT`/`STATEMENT_TASKS`; `Table.cell` (exact rendered cell); cfg `compat_v1` (legacy v1 cases) and `scaled_units`; `_QUOTED_INPUT_RE` check in `validate_example`; v2 features use separate rng streams `question|seed|idx` and `units|seed|idx` so the main rng sequence is unchanged.) Synthetic RAFT dataset -> `dataset.jsonl` + validation. Simulates 12-36 months of a fake startup: P&L, unit economics, pricing plan, cash flow (opening/in/out/closing; an equity raise lands when cash runs low), ARR bridge (open/new/expansion/churned/close), net_revenue series, `ym` (year, month) list. 10 task plans (section 5). Relevant tables + 3-6 distractors (other P&L sections, unit economics, pricing, ARR bridge, quarterly summary, prior-year revenue, headcount, funnel, cash flow, cohorts, inventory, working capital, benchmarks); drops distractors over `max_context_chars`. Ground truth via `common.tools.execute`, checked against independent `common.formulas` truth. Flags `--validate-only --n --output --config`. Missing-data examples are `missing_kind` "cell" (input cell/row removed) or "period" (`period_gap_example`: question asks for a month outside every table, chosen by `absent_months`; share `missing_period_share`). Exports `PROMPT_VERSION`, `TASKS`, `build_example`, `generate`, `validate_example`, `validate_file`, `fy_quarters`, `quarter_values`. |
| `train_qwen3.py` | Unsloth QLoRA SFT; writes `<adapter>/contract.json` ({contract, prompt_version}) after training; renders with Qwen3 template + `TOOLS`; drops > max_seq_length; `train_on_responses_only` (`num_proc=1`); prints masking check; saves adapter + tokenizer + `loss_history.json` + `loss_curve.png`. Refuses stale datasets. `--max-steps --epochs --config --resume` (latest checkpoint in outputs/) `--no-eval` (skip mid-run eval; recommended on 8 GB). |
| `ingest_financials.py` | Excel (2026-10-22 rewrite): `_excel_blocks` (several tables per sheet: any row with >= 2 period cells is a header), `parse_number_text` (Indian/western grouping, (negatives), currency prefixes), `_cell_text`, months-down-rows tables transposed, headerless empty columns dropped, KPI tables don't inherit sheet-wide units (`NON_MONEY_TITLE_RE`), currency fallback, duplicate chunk ids suffixed. Units: Excel normalises `Units:` rows, else title rows / header cells / sheet name; PDF: caption/headers/section, else the section's last 3 text items, else a document-wide units statement (`UNITS_CUE_RE`). Excel (openpyxl values + number formats, pandas structure) and PDF (Docling tables via `export_to_dataframe(doc=doc)`, section headers, text paragraphs ~1,200 chars) -> `TableChunk`s. Detects title rows, `Units:` row, header row, section rows, percent cells, Excel dates -> `Mon-YY`. Long sections split every 12 rows with the header repeated. `PERIOD_RE` matches `Mon-YY`, `Q3 FY26`, `FY27 Plan`, Base/Bull/Bear. |
| `rag_pipeline.py` | `HybridRetriever`: BM25 over full text (finance tokenizer, `normalize_periods`, `QUERY_EXPANSION` incl. "ebitda margin") + dense bge-large over `chunk_summary()` in Chroma (default) or Qdrant + RRF (k=60) + bge-reranker over summaries -> top_k. `load_models=False` = BM25 only. |
| `make_sample_mis.py` | `build_data_lakh()` / `build_workbook(out, lakh=True)` / `--lakh`: P&L + cash in INR lakh (units row "INR (₹ in lakhs)", cash units only in the title), no S&M total row. `sample_startup_mis.xlsx` (Nimbus Botanics, FY26 Apr-25..Mar-26, seed 7; sheets P&L / Unit Economics / Pricing / Cash / Headcount). Module constants `PRICING` (price 649, variable_cost 386.5, fixed_costs 9,850,000, planned_hires 1,420,000) and `PL_SECTIONS`. `build_board_pack_pdf()` / `--pdf PATH` writes the same data as a 3-page board-pack PDF (commentary, 12-month P&L, unit economics) with fpdf2. |
| `run_assistant.py` | End-to-end CLI. Tools shown to the model follow the prompt contract: `FinanceLLM(..., contract=None)` = adapter's contract (`adapter_contract`), `ServerLLM(..., contract=)`, `--contract v1|v2`. `FinanceLLM(adapter_dir or None, base_model)` (None = base model on purpose; `.label`; `.max_time` per-call limit; `.log_rounds` prints tokens/s; `.last_stats`). `ServerLLM(url, model)` for OpenAI-compatible servers (llama-server / Ollama / vLLM / LM Studio): converts messages to OpenAI format with tool_call ids and string arguments, and converts replies back to Qwen3 `<think>` / `<tool_call>` text. `period_guard(question, known_periods, sources)` answers absent-period questions without the model (called in `main()` after relative-period resolution). `run_conversation(llm, user, ...)` = generate -> tool -> generate loop (max 3 rounds) returning `{"thinks","calls","report"}`; `answer()` wraps it. Resolves relative periods before retrieval and prints a NOTE. Flags `--files --question --adapter --top-k --max-new-tokens 2048 --temperature 0.6 --backend local|server --server-url --server-model --retrieval-only --show-context`. |
| `eval_model.py` | Evaluation. `--rescore` re-scores cached run files with the current scorer (no LLM). Real-world = groups per workbook (main + `outputs/sample_startup_mis_lakh.xlsx` built each run; `_realworld_group`). `--v1-compat` rebuilds the cases exactly as the v1 run started them (no row-decimal alignment, data cfg `compat_v1`, no period guard, contract v1); verified to reproduce every stored v1 result. `--contract v1|v2` (default = configured adapter's; base gets the same). `--adapter DIR` evaluates another adapter (e.g. v2). Suites: `synthetic` (fresh examples, seed 9001, `n_per_task` for each task in `configs/eval.yaml` `tasks`, every 7th missing; oracle context) and `realworld` (21 questions from `eval_cases.py`, context from the retriever after period resolution; records retrieval recall). Models: ft / base / both, or `--backend server`. Resumable raw results in `outputs/eval_runs/<run>.jsonl`; report `reports/eval_<date>[_tag].md` (metrics per suite, per-task tables, failure list, retrieval recall). `--gold` = scorer self-test; `--retrieval-only`; `--no-period-guard` (real-world: raw model on absent-period questions; default uses the guard trace). |
| `eval_cases.py` | rw-22..rw-26 multi-period (added 2026-10-22); `build_lakh_cases()` rl-01..rl-05 on the lakh workbook (rl-03/04 require from_lakh); `V1_CASE_IDS` = rw-01..rw-21. `RealCase` list rw-01..rw-21: founder-phrased questions on the sample MIS with Python ground truth from `make_sample_mis.build_data()`. LTV/payback accept GM% from the sheet OR derived from the P&L. 4 missing-period cases (Mar-25, Apr-26, Dec-24, Jun-24); 2 relative-period cases ("last month", "previous month"); `needed_sections` for retrieval recall. |
| `export.py` | Writes `exports/contract.json` + `--contract` hint in SERVING.md. Loads the adapter; `save_pretrained_merged` (`--merged` -> exports/merged_16bit) and `save_pretrained_gguf` (`--gguf q4_k_m ...` -> exports/gguf); writes `exports/gguf/Modelfile` (Ollama) and `exports/SERVING.md` (llama-server / Ollama / vLLM commands). `--name af-finance-qwen3-4b`. |

### `common/` (shared package; prompts/tools/tables changes alter the training contract)
| File | Purpose |
|---|---|
| `formulas.py` | ONLY place finance math lives. Multi-period: `growth_pct(current, prior)` (/|prior|), `average(values)`. Units: `UNIT_FACTORS`, `to_base_units`, `from_lakh`, `from_crore`, `from_thousands`, `from_millions`. (fractions for %). `FinanceMathError` for undefined cases. net_revenue, gross_profit, gross_margin, contribution_margin, cm1, cm2, cm3, margin_pct, ebit_from_net_income, ebitda, adjusted_ebitda, net_burn, runway_months, burn_multiple, cac, ltv, ltv_cac_ratio, payback_period_months, break_even_units, break_even_units_whole, break_even_revenue, magic_number, total, difference, ratio, percent_of. |
| `tools.py` | Safe `finance_calculator` (no eval). `V1_FORMULAS` (the 25 formulas the v1 adapter knows); `formula_signatures(names)`. `REGISTRY`, `_pct` inputs, `@step` refs, list inputs, per-step errors, full precision internally, JSON rounds to 6 dp, display kinds (currency / percent / ratio `x` / months / units). `run_tool_call` never raises; `response_to_json` is the exact tool-message format. |
| `prompts.py` | `CONTRACT` ("v2"), `CONTRACTS`, `tool_description(contract)`, `build_tools(contract)`, `prompt_version(contract)` (v1 = 2013a079fb, v2 = 553efd71f4), `adapter_contract(dir)`; `SYSTEM_PROMPT`, `TOOL_DESCRIPTION`, `TOOLS` (current contract), `build_context`, `build_user_message`, `split_think`, `parse_tool_calls`, `strip_tool_calls`. |
| `tables.py` | `ALIGN_ROW_DECIMALS` module flag (eval `--v1-compat` sets False); `format_cell`, `align_row_decimals` (row-consistent decimals, applied in `render_chunk`), `markdown_table`, `period_span`, `render_chunk`, `render_text_chunk` (shared by generator and ingester). |
| `schemas.py` | CalcStep, CalcRequest, StepResult, CalcResponse, ToolCall, ChunkMetadata, TableChunk. |
| `winperf.py` | `boost_process()`: Windows only; power throttling off + pin to performance cores (fixes E-core slowdowns) + block idle sleep while running. `performance_core_mask()`. |
| `units.py` | `detect_currency` / `first_currency` ("(INR)", "in USD", "Currency: INR"). `detect_units(text)` -> "INR lakh" / "INR crore" / "USD thousands" / "USD millions" ("₹ in lakhs", "Rs. Lakh", "INR Cr", "USD '000", "$000s", "($ in millions)", "in crores"); scaled units only; bare k/m/cr need a currency; lakh/crore only INR. `first_units(texts)`. |
| `audit.py` | `context_numbers`, `literal_numbers`, `unsupported_inputs` (faithfulness). |
| `scoring.py` | `Gold.optional` / `optional_from_steps` (helper intermediates feeding only helpers are optional; 2026-10-22). `ALLOWED_CONSTANTS` = template thresholds 0.5/0.75/14/15/18/24/40/100 + integers 1-12 (month counts; 2026-10-22). `Gold`, `score(trace, gold)`, `aggregate`, `aggregate_by`, `METRICS`, `trace_from_gold_messages`, `required_formulas_from_steps`, `ALTERNATIVES` (cm1 ~ gross_profit ~ contribution_margin etc.), `HELPER_FORMULAS`, `ALLOWED_CONSTANTS` (template thresholds). Numbers in calculator error messages are allowed in reports. A call counts as made even if every step errored. |
| `periods.py` | (2026-10-22: relative phrases only when no explicit period of that kind; grammatical replacement; quarters without a column spelled out as their 3 months.) `resolve_relative_periods(question, periods)`: last/latest/this/current month -> latest `Mon-YY` in the data; previous/prior month -> second latest; same for quarters (`Qn FYyy`). Returns notes. `asked_periods` (strict explicit month/quarter matcher), `quarter_months` (Indian FY), `missing_periods`, `coverage`, `period_not_available_report` (used by the period guard). `month_key`/`quarter_key` sort keys. |

### Config, tests, docs, other
| File | Purpose |
|---|---|
| `configs/data.yaml` | seed 3407, n_examples 1000, test 0.10, missing 0.10 (missing_period_share 0.4), scaled_units 0.2, multi_period weight 1.25, output dataset.jsonl, task_weights (net_revenue 1.0, contribution_margin 1.5, ebitda 1.5, break_even 1.5, ltv_cac 1.5, payback 1.0, runway 1.0, burn_multiple 1.0, magic_number 1.0, gross_margin 0.75), distractors 3-6, max_context_chars 5500, edges (negative_gross_margin .08, seasonal_churn_spike .10, zero_churn .03, capitalized_rnd .25, deferred_revenue .25, step_fixed_costs .25, price_below_variable_cost .06, cash_positive .08, arr_decline .08). Running the generator now produces a v2 (10-task) dataset; the v1 adapter was trained on 6 tasks. |
| `configs/train.yaml` | Qwen3-4B bnb-4bit, max_seq_length 6144, LoRA r32/a32/d0 all linear, 2 epochs, bs 1 x ga 8, lr 2e-4, warmup 10, linear, wd 0.01, adamw_8bit, log 5, eval each epoch, save 50, outputs/, qwen3_4b_finance_lora/. |
| `configs/rag.yaml` | bge-large / bge-reranker-large on CPU, Chroma in-memory, candidates 20, rerank 20, top_k 5, rrf_k 60, 12 rows/chunk, query_expansion true. |
| `configs/train_v2.yaml` | v2 run: dataset_v2.jsonl, outputs/v2, qwen3_4b_finance_lora_v2, 1 epoch, eval off. |
| `configs/eval_v2.yaml` | tasks null (all 11), n_per_task 10. |
| `configs/eval.yaml` | synthetic seed 9001, n_per_task 20, missing_every 7, `tasks` = the 6 v1 tasks, top_k null, temperature 0.0, max_new_tokens 1536, max_time_per_turn 240 s. |
| `tests/test_formulas.py`, `test_tools.py` | Formulas + edge cases; calculator behaviour, parsing. |
| `tests/test_dataset.py` | Generation validates; determinism; all configured tasks present; conversation shape; missing example; period-gap examples; new task families with forced cash_positive / arr_decline validate. |
| `tests/test_ingest_rag.py` | Excel ingestion + BM25 retrieval + audit. |
| `tests/test_scoring.py` | Gold scores 100%; wrong values, hallucinated numbers, a tool call on a missing case and no call are all caught; real-world truth values finite. |
| `tests/test_periods.py` | Relative month/quarter resolution incl. year boundary and unresolvable cases; asked-period phrasings and lookalikes; Indian-FY quarter months; missing periods; guard report. |
| `tests/test_tables.py` | `align_row_decimals`. |
| `tests/test_server_backend.py` | ServerLLM round trip against a fake OpenAI server (tool_call ids, string arguments, reasoning_content); `period_guard`. |
| `tests/test_messy_excel.py` | Number text parsing, currency detection, messy workbook (stacked tables, text numbers, transposed cash, KPI units). |
| `tests/test_units_contract.py` | Unit formulas; calculator conversion before CAC; v1 contract frozen at 2013a079fb; `adapter_contract`; scaled-unit examples validate for all 10 tasks; misquoted input caught; 16 unit phrasings; Excel title units. |
| `tests/test_pdf_ingest.py` | `@slow`: board-pack PDF via Docling equals Excel chunks (Revenue, Variable Costs, Profit Summary incl. negatives), commentary text chunk, churn % row. ~2.5 min. `pytest -m slow`. |
| `pyproject.toml` | Deps + extras; pytest `-m 'not slow'` by default. |
| `README.md` | Overview, setup, commands for every step (data, train, ingest, assistant, eval, export, server mode), dataset design, evaluation, layout. |
| `CLAUDE.md` | Session-start + documentation rules, architecture rules, definitions, layout, build status. |
| `.gitignore` | venvs, outputs/, *_lora/, weights, unsloth_compiled_cache/, dataset*.jsonl, vector stores, data/private/, xlsx/xls/csv/pdf except sample_startup_mis.xlsx, secrets, exports/. |
| `.gitattributes` | `*.sh text eol=lf`. |
| `MODEL_CARD.md` | (2026-10-22: v2 data capabilities, units / contract limitations, retrieval 31/31; eval numbers pending.) Draft model card: summary, intended use, out of scope, training data, evaluation (to fill from reports/), limitations, disclaimer, privacy, component licenses (Qwen3 Apache 2.0, Unsloth Apache 2.0, BGE MIT, Docling MIT, llama.cpp MIT; project license not chosen). |
| `.github/workflows/tests.yml` | CI on push/PR: ubuntu, Python 3.12, `pip install pydantic pyyaml pandas openpyxl rank-bm25 pytest`, `pytest -q`, then generate + validate 300 examples. Verified locally in a clean Python 3.14 venv (50 passed). |
| `.claude/settings.json` + `.claude/hooks/` | SessionStart (inject memory + changelog) and Stop (block if files newer than docs) hooks. The parent workspace has its own `.claude/settings.json` calling the same scripts. Hooks may need `/hooks` or a restart to activate in the session that created them. |
| `documentation/memory.md`, `Changelog.md` | This file; change log (newest first). |
| `reports/` | Evaluation reports (committed). Not created yet. |
| Ignored artifacts | `dataset_v2.jsonl` (v2, 3,000 ex, 11 tasks), `dataset.jsonl` (v1, 1,000 ex), `outputs/` (checkpoints, `train_log.txt`, `eval_runs/`), `qwen3_4b_finance_lora/`, `exports/`, `unsloth_compiled_cache/`. |

---

## 5. Dataset contract (do not change casually)

- Row: `{"id", "messages", "meta"}`. messages: system -> user (`<financial_context>` + `Question:`) -> assistant (`<think>...</think>\n\n` + one `finance_calculator` tool call with `{"steps": [...]}`) -> tool (`response_to_json`) -> assistant (`<think>` check + Markdown report). Missing-data: system -> user -> one assistant turn naming the missing input, no tool call.
- meta: id, task, variant, split, period, company, currency, edge_cases, missing, prompt_version, truth (step id -> value | null = expected error).
- `TOOLS` are not stored per row; changing `SYSTEM_PROMPT`/`TOOLS` changes `PROMPT_VERSION` -> regenerate + retrain.
- Tasks/variants: net_revenue (basic/deferred); contribution_margin (cm2/cm3); ebitda (from_ebit/from_net_income, +adjusted, +margin); break_even (simple/components/step_fixed/scenarios, +impossible); ltv_cac + payback (S&M given vs summed; GM given vs from P&L); runway (net_burn -> runway_months; cash_positive -> runway error); burn_multiple (bridge: total(new, expansion) - churned, or open_close: closing - opening; arr_decline -> error); magic_number (Indian-FY quarters Q1 = Apr-Jun from `fy_quarters`, prior-quarter S&M; falls back to runway if < 2 quarters); gross_margin (net_revenue, gross_profit, gross_margin).
- Question period phrasing (`say_period`): training questions write the period as `Mar-26` (2/6), `March 2026`, `Mar 2026`, or `Mar '26`; quarters as `Q4 FY26` (2/4), `Q4 of FY26`, `Q4 FY2026`. Tables and `<think>` always use the table label. Never `April 25` (reads like a date). `rag_pipeline.normalize_periods` maps all of these back for BM25.
- Period-gap missing examples (since 22:55): the question asks for a month outside every table (often the same month a year earlier); answer names the month and the covered range, no tool call; meta `missing` = the month, `missing_kind: period`. These are the only missing examples for runway / burn multiple.
- Missing-data policy: only drop inputs that cannot be derived from other rows. Runway and burn multiple have NO cell-type missing examples (cash in = closing - opening + out; net new ARR = close - open). Missing examples also exclude distractors that could supply the value (other P&L sections, unit economics, cohorts, prior year, ARR, quarterly).
- Report format: `## CFO Report: <title> (<period>)`, company/source/currency line, `| Metric | Value | Formula |`, `**Inputs used:**`, `**CFO take:**` (thresholds: leakage 5-12%, CM2 15%, LTV:CAC 1x/3x, payback 12/24 mo, runway 6/12/18 mo, burn multiple 1/2/3, magic number 0.5/0.75, gross margin 40%).
- v1 stats (dataset.jsonl, the adapter is trained on this): 1,000 ex (900/100), net_revenue 115/15, contribution_margin 170/19, ebitda 153/20, break_even 161/16, ltv_cac 193/18, payback 108/12; 89 missing; tokens median 3,613, p90 4,817, max 5,821.
- v2 trial (1,500 ex, scratchpad only): all 10 tasks, 117 missing, 0 mismatches.
- Validation: re-run tool == stored tool message; tool == independent truth; literal inputs in context; result displays in report.
- v2 additions (2026-10-22): questions reworded by `vary_question` (40% extra template, 15% lower-case first word, 15% no '?', 20% prefix, 15% suffix; period label kept). Quarterly-summary distractor gets a `<think>` line "[k] has quarter totals ...; I need <month> monthly figures". 20% of companies report statements (P&L, cash, ARR, quarterly, working capital, one-offs) in INR lakh / INR crore / USD thousands with 2 decimals; unit economics (counts, ARPU, S&M row, %) and pricing stay plain. Statement tasks: `<think>` units note; report header `Currency: INR lakh`. LTV:CAC / payback with S&M summed from a scaled P&L: steps `sm_spend_<unit>` = total -> `sm_spend` = from_<unit> -> cac; report shows both S&M rows. Quoted inputs in `<think>` = exact rendered cell. Validation also checks every quoted input is a cell of its row. A missing-data example never deletes a row another input needs (cell blanked instead).
- Multi-period (`multi_period`, variant `<mom|range_total|average|ytd|qoq>:<metric>`; qoq 20% when >= 2 fiscal quarters, reads the quarterly summary): flows (gross revenue, COGS, performance marketing, payroll, new customers) can be summed/averaged; stocks (closing ARR, closing cash, active customers) only MoM. Inputs named `<label> (<month>)`. Question names only the end period ("the 4 months ending Mar-26", "YTD ... through Mar-26"); `<think>` lists the exact months used. YTD = Indian FY from April.
- v2 trial (3,000 ex, scratchpad; after multi-period: 286 multi_period): 0 mismatches; 277 missing (163 cell, 114 period); 620 scaled (302 USD thousands, 262 INR lakh, 52 INR crore); 88 conversion examples (51 LTV:CAC, 37 payback); quarterly note in 539 examples.

---

## 6. Results so far

- Tests: 58 fast + 1 slow pass (22:55).
- Smoke training (5 steps): eval_loss 0.8986, train_loss 1.12, peak VRAM 5.64 GB.
- Masking check: 327/1,826 tokens of example 0 carry loss, starting at `<think>`.
- Full v1 training loss: 1.121 (step 5), 0.711 (10), 0.298 (20), 0.128 (30), 0.064 (40), 0.023 (50), 0.015 (80), 0.013-0.014 (95-100). **Epoch-1 eval loss 0.0128** (100 held-out examples, 168 s) - same as train loss, no overfitting gap. Run 1 then stalled (steps 113->117 took ~1 h, VRAM spill); log kept at `outputs/train_log_run1_stalled.txt`. Resumed from checkpoint-100 (global_step 100, epoch 0.889) with no mid-run eval; new log `outputs/train_log.txt`. Final: loss 0.0097 (step 215), 0.0090 (220), 0.0093 (225); runtime 98.8 min for the resumed part; peak VRAM 5.90 GB.
- Scorer self-test (gold answers, 360 synthetic cases): 100% on every metric.
- Retrieval recall (21 real-world questions, top-5): 100% (98.2% before adding the "ebitda margin" expansion).
- **Demo with v1 adapter (18:20 IST, `outputs/demo_run.txt`)**: CM1/CM2 Mar-26 (net revenue 20,647,900; CM2 9,126,370 = 44.20%), LTV:CAC Jan-26 (CAC 2,262.81; LTV 11,531.08; 5.10x), EBITDA Mar-26 (-2,077,150; added back capitalized R&D amortization; also volunteered EBITDA margin -10.06% because the Revenue chunk was retrieved), break-even 37,524 -> 42,934 units after hires. All correct vs Python, all inputs faithful. Latency 64-92 s/question (local HF generate, temperature 0.6). Flaw: every report header said "Sample Startup" - the model derives a company name from the source FILE NAME (sample_startup_mis.xlsx -> "Sample Startup"; tandem_cloud_monthly_mis.xlsx -> "Tandem Cloud") because v1 training reports printed the company name next to the source file. Generator fixed for v2 (header = source + currency only).
- **Eval, v1 adapter, fine-tuned half COMPLETE (22:50 IST, `outputs/eval_runs/fine_tuned_v1.jsonl`): 70/81 end-to-end.** Synthetic 54/60 (normal 49/54, missing 5/6). Real-world 16/21: normal 16/17, missing-period 0/4. Real-world failures: (d) rw-16..19 (question asks for a period NOT in the MIS: Mar-25, Apr-26, Dec-24, Jun-24): the model silently used another period's column (rw-16 asked Mar-25, used Mar-26 values, labelled the report "Mar-25") - serious; v1 training only had missing ROWS, never missing PERIODS. (e) rw-12: took ARPU from the Aug-25 column (1,240.04) instead of Sep-25 (1,240): off-by-one column; the Sep-25 cell renders as `1,240` while the rest of the row has 2 decimals (ingestion `format_cell` drops trailing .00 per cell) - possible trigger. Scorer verified correct on both.
- Earlier interim note (65/81): synthetic 54/60; real-world 5/5 so far (rw-01..rw-05). Failures: (a) false "missing data" refusals with the data present: raft-1020001 (adjusted EBITDA + margin; revenue table was in context), raft-1050007 (payback); (b) raft-1020006 ebitda missing-data case where the refusal did not clearly name the missing input; (c) LTV:CAC when S&M must be summed from marketing rows: raft-1040001/04/07 took "S&M Spend" from the QUARTERLY summary distractor (a v2 table v1 never saw) and cited a non-existent monthly column. Eval contexts come from the current (v2) generator: new distractors (ARR, quarterly, cash flow) and varied period phrasing, so these scores measure generalisation.
- 2026-10-22: `--v1-compat` re-score check: 81/81 fine-tuned + every cached base row reproduce exactly. Base reruns of time-limited cases: raft-1020005 and raft-1030003 OK (raft-1030003 was OK before too), others still FAIL (no tool call / wrong values).
- **v1 eval FINAL (2026-10-22 11:51, `reports/eval_2026-10-22_v1.md`)**: synthetic end-to-end 90.0% ft vs 15.0% base (numeric 90.7/9.3, formula 96.3/9.3, faithful 100/40.7, missing 83.3/66.7, false refusals 3.7/59.3); real-world 76.2% vs 9.5% (raw, no guard; ft fails rw-12 + rw-16..19). Per task (ft): break_even/CM/net_revenue 100%, payback 90%, ebitda 80%, ltv_cac 70%.
- PDF: board-pack PDF via Docling -> 9 chunks; P&L sections identical to Excel incl. negative EBIT; percent rows intact; commentary as a text chunk. First run downloads Docling models; ~2.5 min on CPU.

---

## 7. Decisions and why

- Qwen3-4B (fits 8 GB). Hybrid layout (user's choice). Template reasoning, no LLM API (determinism, ground truth).
- Tool calling with multi-step `steps` + `@refs`; `_pct` inputs; display strings copied by the model.
- 3-12 month windows; `max_context_chars` 5500 + `max_seq_length` 6144 (digit-by-digit tokenization).
- Dense + reranker on digit-free summaries (the cross-encoder mis-ranked numeric tables).
- Retrieval models on CPU at inference.
- Relative periods resolved by deterministic preprocessing, not training: "last month" = latest month in the MIS.
- Evaluation: synthetic with oracle context (model skill) + real-world through RAG (system skill); fresh seed 9001; greedy decoding; ground truth always from Python.
- `configs/eval.yaml` `tasks` must match what the evaluated adapter was trained on.
- Server backend (OpenAI-compatible) so users can run the GGUF with llama-server / Ollama without the training stack; the same pipeline and the same evaluator work through it.
- No missing-data examples where the value is derivable (would teach wrong refusals).
- The fake "Cash Position" distractor was replaced by a consistent cash-flow table from the simulated data.
- Report headers contain only facts present in the context (`Source: <file> | Currency: <cur>`); no company name (v1 adapter guesses one from the file name).
- Prompt contracts (2026-10-22): the tool description lists the formulas, so adding formulas changes what a trained model sees. Each adapter records its contract (`contract.json`); old adapters = v1; inference and eval show the matching tool list. The calculator itself always accepts every formula.
- Evaluation reproducibility: a resumed eval must rebuild identical cases; generator changes are gated (`compat_v1`) and new random choices use separate rng streams so old seeds keep producing the same examples.
- Lakh/crore: amounts are passed to the calculator as shown and results stay in the table's units; conversion (`from_lakh` etc.) only when scaled amounts meet counts or plain-unit values (CAC). The model never multiplies by 100,000 itself.
- Units detection only for scaled units, from table-specific text first; narrative text needs a units cue word so "2 mn users" is never read as units.

---

## 8. Known gaps / risks

- Model quality not measured yet (eval pending). Training loss ~0.01 means the templates are learned; real-world generalisation is the open question.
- The v1 adapter does not know runway / burn multiple / magic number / standalone gross margin (v2 data ready, needs retraining).
- Phrasing variety in training questions is limited (3-4 templates per task); the real-world eval uses different phrasing on purpose.
- Multi-period added to data (not trained); no QoQ from quarterly tables, no multi-metric (e.g. average burn = average of out - in). Lakh/crore handled in data + ingestion but not trained yet and untested on a real lakh-format file; break-even with fixed costs in lakh (pricing sheet) is not generated.
- Messy real MIS layouts (merged cells, transposed tables, several tables per sheet) untested; only synthetic files tested.
- Export/GGUF and the llama-server path not run yet; quantization loss unknown.
- No UI/API; models load per run.
- Release still needs: project license choice (owner), eval numbers in MODEL_CARD.md, versioning/tags, packaged install, HF upload.

---

## 9. User preferences and standing instructions

- Show a **build progress %** after every build session (section 1).
- Keep `documentation/memory.md` and `documentation/Changelog.md` updated after every prompt, automatically. Log every change with date and time (IST).
- New sessions: read these two files instead of analysing the codebase.
- Git: no "Co-Authored-By: Claude" trailer. The user commits on their own. Do not push unless asked.
- Wants clarifying questions when needed; recommendations over option lists; "continue building" = keep working through the milestones in order.
- Other project: AFSI Suite (drone swarm OS) in `Desktop/aeryal flow/afsi suite/` (unrelated).

---

## 10. Commands cheat sheet

```bash
PY=/c/Users/nevin/miniconda3/envs/unsloth_env/python.exe
cd "/c/Users/nevin/Desktop/aeryal flow/finance llm/af-finance-llm"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q -p no:cacheprovider          # fast tests
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q -p no:cacheprovider -m slow  # Docling PDF test
$PY generate_raft_data.py                  # NOTE: now writes a v2 (10-task) dataset
$PY train_qwen3.py --max-steps 5           # smoke test
$PY -u train_qwen3.py > outputs/train_log.txt 2>&1
$PY make_sample_mis.py --pdf sample_board_pack.pdf
$PY ingest_financials.py sample_startup_mis.xlsx
CUDA_VISIBLE_DEVICES="" $PY run_assistant.py --retrieval-only
$PY run_assistant.py                       # GPU must be free
$PY eval_model.py --gold                   # scorer self-test
CUDA_VISIBLE_DEVICES="" $PY eval_model.py --retrieval-only
$PY eval_model.py                          # ft + base, both suites
$PY export.py --gguf q4_k_m
~/.unsloth/llama.cpp/build/bin/Release/llama-server.exe -m exports/gguf/<file>.gguf --jinja -ngl 99 -c 8192 --port 8080 --alias af-finance-qwen3-4b
$PY eval_model.py --backend server --tag gguf
```

---

## 11. Data sizing guidance (advice given to user 2026-10-21)

- Test-set size changes how much you can TRUST the score, not the score itself. Training data changes the score.
- v1 test split: 100 examples total, only 12-20 per task. At ~90% accuracy the 95% margin of error is about +/-6 points overall and +/-13-17 points per task, so per-task scores are mostly noise.
- Plan: a separate, larger eval set (target ~100 per task family + ~100 missing-data = ~700; generation is free and takes seconds) with a DIFFERENT seed and ideally held-out wording. `eval_model.py` now does this (seed 9001; raise `n_per_task` in `configs/eval.yaml`).
- Hand-written "real-world" set: 30-50 questions on the sample MIS and later real anonymised MIS files. `eval_cases.py` has 21 so far.
- Training data: do not just add more of the same templates. Loss hit ~0.02 at 0.44 epochs, so the templates are already learned. Gains come from diversity: question phrasings, table layouts, label synonyms, new task families (now added: runway, burn multiple, magic number, gross margin), multi-period. Target ~2,500-3,000 diverse examples, 1 epoch (~4-5 h on the RTX 5050).
- Order: finish training -> run eval -> read per-task errors -> then decide what training data to add.
