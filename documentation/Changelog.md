# Changelog: af-finance-llm

Every change to code, data, config, environment or models, newest first. Times are IST (UTC+05:30).
Format: `## YYYY-MM-DD HH:MM IST - title`, then what changed, which files, and why.
Entries before 14:09 on 2026-10-21 were reconstructed from file timestamps, git history and the session transcript.

---

## 2026-10-23 20:31 IST - Model comparison from online sources; Claude API eval backend
- User asked to compare the model with finance-strong models. First plan (run Fin-R1 7B, Qwen3-8B, Gemma 4 E4B GGUFs locally + frontier APIs) was started, then **stopped on the user's instruction: no model downloads on the laptop, use online sources only**. Download killed after 379 MB of Fin-R1; `C:\Users\nevin\.local\gguf-models\` deleted (nothing kept). No API keys are set (ANTHROPIC/OPENAI/GEMINI/GOOGLE/OPENROUTER all unset; no `ant` CLI), so no frontier model was run either.
- New `reports/model_comparison_2026-10-23.md`: (A) our harness results (q8_0 GGUF 95.5% / 96.8%, adapter, v1, base Qwen3-4B 9.1% / 12.9%; frontier not run) kept separate from (B) published benchmarks: Vals AI Finance Agent v2.1 (2026-10-22: Claude Opus 5.5 67.85%, Gemini 4 Argon 67.66%, Claude Sonnet 5.5 67.28%, Claude Haiku 5.5 65.37%, GPT-6.1 Sol 61.82% ...), AIMultiple FinanceReasoning-hard (Opus 5.5 91.6%, Gemini 4 Argon 90.76%, GPT-5.6 Sol 90.34%, Haiku 5.5 85.71%, GPT-OSS-120B 81.09%, Llama 4 Maverick 75.21%, Mistral Small 3.2 24B 41.18%), Fin-R1 paper FinQA / ConvFinQA (Fin-R1 7B 76 / 85, DeepSeek-R1 71 / 82, Qwen2.5-7B 60 / 66), GPT-4o FinQA 72.5 (Fino1 leaderboard), FinanceBench aggregators (unverified). States plainly that A and B are not comparable and our model's scope is narrower.
- New `common/claude_llm.py` `ClaudeLLM` (official Anthropic SDK; same interface as FinanceLLM / ServerLLM; same system prompt + finance_calculator tool converted by `anthropic_tools`; adaptive thinking with summarized display, `output_config.effort` (default high), no temperature; raw assistant content blocks replayed unchanged per conversation; all tool results of a turn in one user message; refusal handling; server-side fallback `fallbacks="default"` + beta `server-side-fallback-2026-07-01` for Opus / Sonnet (none for Haiku), a fallback-served answer is tagged "[served by ...]" in the think trace; usage counters). Not run yet (no key).
- `eval_model.py`: `--backend claude --claude-model (claude-opus-5-5) --claude-effort (high)` -> run file `outputs/eval_runs/claude_<model><tag>.jsonl`, also in `--rescore`; `--server-api-key-env VAR` (bearer key for hosted OpenAI-compatible APIs). Resume now **retries rows whose report is "(generation error ...)"** (transport / server failures such as today's llama-server OOM) instead of keeping them as model failures; `--rescore` keeps the latest non-error row per case. `import os`.
- `run_assistant.py` `ServerLLM(..., api_key=None)`: optional `Authorization: Bearer` header.
- `pyproject.toml`: extra `compare = ["anthropic>=2.2"]`; `anthropic 2.2.0` installed in unsloth_env.
- `tests/test_claude_backend.py` (new; skips without anthropic): tool list equals the fine-tuned contract; fake-client round trip (no temperature sent, raw blocks replayed, tool_result id). 105 fast tests pass.

## 2026-10-23 18:46 IST - q8_0 GGUF (dequantized merge) matches the adapter; milestones 6 + 7 done
- dq q8_0 eval finished (exit 0, 0 time-limit hits), report `reports/eval_2026-10-23_gguf_dq8.md`, run file `outputs/eval_runs/server_af-finance-qwen3-4b_gguf_dq8.jsonl`: **synthetic 95.5%, real-world 96.8% (30/31), faithfulness 100% / 100%, numeric 97.0 / 96.3, missing 81.8 / 100, false refusals 1.0 / 3.7** = identical to the adapter (`eval_2026-10-23_v2.md`), at **16 s / 15 s per case vs 60 s**. Case-level: dq8 passes raft-1020001 (adapter fails), fails raft-1100004 (adapter passes); everything else the same (other flips in a naive diff are old-scorer values stored in `fine_tuned_v2.jsonl`).
- llama-server stopped; GPU free.
- `MODEL_CARD.md`: new "Quantized GGUF (llama.cpp)" section (table: adapter / shipped q8_0 dq / Unsloth q4_k_m; merge explanation; no 4-bit GGUF yet); "Not yet measured" line replaced.
- README export section: new `export.py` behaviour (dequantized merge, q8_0 default, eval other types first), serve command with `-np 1 -fa on`, `--contract v2` on run_assistant / eval.
- Build progress 90% -> 93% (milestone 6: 13 -> 15, milestone 7: 7 -> 10).
- Tests: 103 fast pass (incl. the 2 new export tests).

## 2026-10-23 18:20 IST - Fix confirmed; export.py rewritten around the dequantized merge; exports rebuilt
- dq q8_0 eval (`--tag gguf_dq8`): first 20 cases **20/20** (net revenue 10/10, contribution margin 10/10 incl. both missing-data cases); Unsloth-merge q4_k_m got 9/20 and Unsloth-merge q8_0 10/16 on the same cases. ~15 s (net revenue) - 22 s (CM) per case. Full run continuing.
- `export.py` rewritten (no Unsloth export path any more): `merge_dequant(adapter, base_model, out)` (bitsandbytes `dequantize_4bit` per nf4 tensor + `merge_tensor` W + (alpha/r) B@A in fp32 -> bf16; tensor names from the base checkpoint; ~4.5 GB shards + index; config without `quantization_config`; tokenizer / chat template copied from the adapter dir; exits if the number of merged LoRA modules != adapter modules) -> `exports/merged_16bit/`; `to_gguf(...)`: llama.cpp `convert_hf_to_gguf.py --outtype bf16` (run with CUDA hidden) -> `llama-quantize` per type -> `exports/gguf/<name>.<TYPE>.gguf`, bf16 deleted unless `--keep-bf16`; `--gguf` default **q8_0** (the only verified type); `--llama-cpp DIR` (default `~/.unsloth/llama.cpp`), `--device cpu`; `write_modelfile` with the Qwen3 Ollama tool template built in (`OLLAMA_TEMPLATE`, from Unsloth's Modelfile) + `num_ctx 8192`; `--finalize-only` rewrites Modelfile / SERVING.md / contract.json. SERVING.md: `-np 1 -fa on`, GPU-build note, VRAM note, `--contract` on every server command. `collect_gguf` (Unsloth `_gguf` dir handling) removed.
- `tests/test_export.py` (new; skips without torch / bitsandbytes, so CI skips it): tiny fake nf4 checkpoint + adapter -> merged weight equals deq(nf4) + scale B@A exactly and differs from the original-weight merge; norm untouched; no quant stats; config cleaned; Modelfile has the tool template. 2 passed.
- `exports/gguf/` (Unsloth merge, bad) moved to `exports/old_unsloth_merge/` (not deleted). Running `python export.py` (log `outputs/export_dq_log.txt`) to rebuild `exports/` with the new code.
- 18:18 `python export.py` done (exit 0, ~3 min on CPU): `exports/merged_16bit/` (2 shards, config without quantization_config, bf16, tied embeddings, tokenizer + chat_template from the adapter) is **bit-identical** to the scratch `exports/merged_dequant/` (398/398 tensors equal); `exports/gguf/af-finance-qwen3-4b.Q8_0.gguf` (4,280,405,472 bytes, same as the evaluated `qwen3-4b-dq.Q8_0.gguf`), `exports/gguf/Modelfile`, `exports/SERVING.md`, `exports/contract.json` (v2).

## 2026-10-23 18:09 IST - Root cause: the Unsloth merge; dequantized-nf4 merge; dq GGUF evals
- q8_0 (Unsloth merge) eval: llama-server died at ~case 17 with `CUDA error: out of memory` (q8_0 + buffers ~5.4 GB on the 8 GB card); later rows failed in 4 s with WinError 10061/10054 (connection refused / reset). Eval stopped; 9 crash rows removed from `outputs/eval_runs/server_af-finance-qwen3-4b_gguf_q8.jsonl` (backup `.with_crash_rows`), 16 valid rows kept: net revenue 10/10, contribution margin 1/6 (same margin_pct drop as q4). So the near-lossless q8_0 shows the same regression -> **the cause is the merge, not q4 quantization**.
- Why: QLoRA trains on y = x @ deq(nf4(W)) + x @ BA. Unsloth's `save_pretrained_merged` / `save_pretrained_gguf` merges BA into the ORIGINAL bf16 `unsloth/Qwen3-4B` weights, i.e. it adds back the nf4 rounding error the adapter never saw.
- Scratchpad `merge_dequant.py`: W' = deq(nf4(W)) + (alpha/r) * B @ A from `unsloth/Qwen3-4B-unsloth-bnb-4bit` (bitsandbytes `QuantState.from_dict` + `dequantize_4bit`; double-quant nf4; Unsloth keeps layers 0, 1, 3, 4, 6, 33, 34 attn/mlp in 16-bit) + adapter `qwen3_4b_finance_lora_v2`, same tensor names / shard layout / config / tokenizer as `exports/gguf`, written to `exports/merged_dequant/` (CPU, ~3 min). Self-check: 210 nf4 tensors dequantized, LoRA merged into 252; on never-quantized layers max abs diff vs Unsloth's merge = **0** (LoRA math matches); on quantized layers mean relative diff **9.3%** (the size of the gap the bf16 merge introduced).
- Running: `convert_hf_to_gguf.py exports/merged_dequant --outtype bf16` -> `llama-quantize ... q4_k_m` -> `exports/merged_dequant/qwen3-4b-dq.Q4_K_M.gguf` (log `outputs/convert_dequant_log.txt`). Then serve it and rerun the eval with `--tag gguf_dq`.
- 18:04 done (exit 0): `exports/merged_dequant/qwen3-4b-dq.BF16.gguf` (8.05 GB) and `qwen3-4b-dq.Q4_K_M.gguf` (2.50 GB; llama-quantize 32 s). Served on the GPU (3.7 GB VRAM, log `outputs/llama_server_dq_log.txt`); eval `--tag gguf_dq` running (log `outputs/eval_gguf_dq_log.txt`).
- dq q4_k_m is WORSE: 0/5 net revenue (Unsloth-merge q4: 9/10), e.g. raft-1000000 adds an unrelated `ebit_from_net_income` step with invented inputs (-12.5). Likely double rounding: dequantized nf4 values sit on the nf4 grid, and re-quantizing them onto q4_k's different 4-bit grid adds a second full quantization error. Stopped after 5 cases (run file `server_af-finance-qwen3-4b_gguf_dq.jsonl`, 5 rows).
- 18:07 `llama-quantize ... q8_0` -> `exports/merged_dequant/qwen3-4b-dq.Q8_0.gguf` (4.28 GB). Served with `-fa on -b 512 -ub 256` to cut compute buffers (5.36 GB VRAM, log `outputs/llama_server_dq8_log.txt`); eval `--tag gguf_dq8` running (log `outputs/eval_gguf_dq8_log.txt`). This is the closest GGUF to the training-time model.

## 2026-10-23 17:54 IST - q4_k_m GGUF eval result; q8_0 eval started
- q4_k_m run finished (exit 0), report `reports/eval_2026-10-23_gguf.md`, run file `outputs/eval_runs/server_af-finance-qwen3-4b_gguf.jsonl` (141 rows). vs the adapter (`eval_2026-10-23_v2.md`): **synthetic 81.8% vs 95.5%**, numeric 82.8 vs 97.0, faithful 92.9 vs 100, missing 72.7 vs 81.8, false refusals 0 vs 1.0; **real-world 87.1% (27/31) vs 96.8%**, faithful 92.6 vs 100. Latency **14 s vs 60 s per case**.
- q4 failure pattern: drops `margin_pct` / `@ref` steps and types self-computed numbers into later steps or the report (all 10 synthetic contribution-margin cases, rw-03, rw-05, rw-08 EBITDA margin, gross margin), LTV / LTV:CAC chain errors (raft-1040000/03/07, rw-09), extra missing-data tool calls (raft-1080006). These break the "model never does arithmetic" rule, so q4_k_m is not shippable as is.
- 17:52 llama-server restarted on `Qwen3-4B.Q8_0.gguf` (5.4 GB VRAM, `outputs/llama_server_q8_log.txt`); `eval_model.py --backend server --contract v2 --config configs/eval_v2.yaml --tag gguf_q8` running (log `outputs/eval_gguf_q8_log.txt`).

## 2026-10-23 17:26 IST - Resumed: GPU llama.cpp build for the GGUF eval; q4 regression; q8_0 diagnostic
- User said continue. Chose option (b): official llama.cpp release **b11734** (same version as Unsloth's converter that wrote the GGUF), Windows **CUDA 13.4** x64 build + cudart zip, into its own folder `C:\Users\nevin\.local\llama.cpp\b11734-cuda13\` (outside the repo). CUDA 12.4 build rejected: no native sm_120 (Blackwell RTX 5050). Driver 610.47 = CUDA 13.3 (minor-version compatible). Fallback if it fails: the Vulkan build. Checksums verified against the GitHub release digests (llama 16ebb234..., cudart 738f8c25...).
- Extracted to `...\b11734-cuda13\bin\`: `llama-server --list-devices` -> `CUDA0: NVIDIA GeForce RTX 5050 Laptop GPU`. Server started with `-ngl 99 -c 8192 -np 1` (one slot: the eval is sequential, keeps the full 8,192 context per request); GPU memory 3.7 GB, model loaded in ~3 s.
- GGUF eval restarted against it (same command, log `outputs/eval_gguf_log.txt` after a RESUMED marker). Speed: 12-18 s per case (local HF adapter: ~60 s).
- Early q4_k_m regression: raft-1000009 (leakage from the wrong rows, typed a self-computed difference into the next step) and **every contribution-margin case** (raft-1010000..05) drop the `cm*_pct` margin_pct steps, then write a self-computed % in the report (46.72% vs the tool's 46.71%); raft-1010000's think also has a stray capitalized-R&D note. The adapter passes all of these.
- Template ruled out: scratchpad `template_diff.py` renders raft-1010000 via llama-server `/apply-template` and via the HF tokenizer chat template: identical (8,735 chars). So the cause is the weights: the merge (LoRA trained on the nf4 base, merged into bf16 `unsloth/Qwen3-4B`) or the q4_k_m quantization.
- Diagnostic started: `convert_hf_to_gguf.py exports/gguf --outtype q8_0 --outfile exports/gguf/Qwen3-4B.Q8_0.gguf` (from the merged shards, log `outputs/convert_q8_log.txt`). If q8_0 passes the CM cases -> q4 quantization is the cause; if it fails too -> the merge.
- 17:27 q8_0 written: `exports/gguf/Qwen3-4B.Q8_0.gguf` (4.28 GB, 398 tensors; convert exit 0). Plan: let the q4 run finish (10/10 contribution-margin cases FAIL the same way), then serve q8_0 and run `--tag gguf_q8`.

## 2026-10-23 15:19 IST - Paused by the user
- Stopped the GGUF eval (0 of 141 cases finished; run file empty), llama-server and the watcher. No project process running, GPU free. Resume steps in memory.md section 1.
- Build progress 87% -> 90% (milestone 7: 4 -> 7, GGUF exported and loads in llama-server).

## 2026-10-23 15:17 IST - GGUF export done; export.py --finalize-only; GGUF eval started (CPU llama-server)
- Export finished 15:14 (exit 0): `exports/gguf/Qwen3-4B.Q4_K_M.gguf` (2.50 GB; Unsloth deleted the bf16 intermediate), merged 16-bit HF shards in `exports/gguf/` (8 GB, needed only for re-conversion / vLLM), `exports/contract.json` = v2 / 5c6b089a25 / qwen3_4b_finance_lora_v2.
- `export.py`: GGUF handling moved into `collect_gguf(target, quant)`: moves `*.gguf` from Unsloth's `<target>_gguf/` into `exports/gguf/`; keeps Unsloth's own Modelfile (Qwen3 tool-calling TEMPLATE + stop tokens, which Ollama needs; BOM-tolerant read, `FROM ./<gguf>`, `num_ctx 8192` appended), else the minimal one; `write_serving(...)`; new `--finalize-only` (no model load: re-gather files, rewrite Modelfile / SERVING.md / contract.json). First version deleted Unsloth's Modelfile before checking it and fell back to the minimal one; the check is now done first and the file was restored from its printed copy. Ran `--finalize-only`: Modelfile + SERVING.md now name `Qwen3-4B.Q4_K_M.gguf`.
- llama-server (`outputs/llama_server_log.txt`) serves the GGUF on :8080 (alias af-finance-qwen3-4b, `--jinja -c 8192`, 4 slots). Found: the Unsloth llama.cpp build is **CPU-only** (no CUDA backend), so it runs on the CPU. memory.md section 3 corrected.
- Started `eval_model.py --backend server --contract v2 --config configs/eval_v2.yaml --tag gguf` (log `outputs/eval_gguf_log.txt`, run file `outputs/eval_runs/server_af-finance-qwen3-4b_gguf.jsonl`).

## 2026-10-23 15:13 IST - GGUF export in progress (v2); export.py output-dir fix
- `export.py --gguf q4_k_m` (started 14:15, log `outputs/export_v2_log.txt`): Unsloth downloaded `unsloth/Qwen3-4B` (16-bit, 7.5 GB, ~50 min at ~2.5 MB/s) into the HF cache (later exports skip it), then merged the v2 adapter into 16-bit weights in `exports/gguf/` (`model-0000{1,2}-of-00002.safetensors` 4.7 + 2.9 GB, config, tokenizer, chat_template.jinja, a `.cache/huggingface/` download cache). These are the intermediate HF-format files for llama.cpp conversion; the `.gguf` file comes next. `exports/` is git-ignored.
- 15:13 bf16 GGUF written (`Qwen3-4B.BF16.gguf`, 8.05 GB), q4_k_m quantization running (`Qwen3-4B.Q4_K_M.gguf`). Unsloth 2026.10 writes them to `exports/gguf_gguf/` (sibling `<target>_gguf` dir), not `exports/gguf/`, so the running `export.py` would find no .gguf and write `<model>.gguf` placeholders into the Modelfile / SERVING.md.
- `export.py` fixed: after `save_pretrained_gguf`, moves `*.gguf` from `<target>_gguf/` into the target dir and removes the empty sibling. The running process uses the old code; its files will be moved by hand and Modelfile / SERVING.md regenerated.

## 2026-10-23 14:14 IST - v2 eval complete; v2 is the default adapter; training overwrite guard
- v2 eval finished ~14:10 (exit 0; harmless resource_tracker traceback). Base model: 136 cases at ~100-110 s each, 0 time-limit hits; generation 11-16 tok/s, no VRAM spill. Run files: `outputs/eval_runs/fine_tuned_v2.jsonl` and `base_v2.jsonl`, 141 rows each.
- Final report `reports/eval_2026-10-23_v2.md` (rewritten by `--rescore --models both` with the current scorer; the in-process run had the old scorer: 7 fine-tuned rows flip to pass = 6 burn-multiple closing-opening + rw-24).
- Results, fine-tuned vs base: **synthetic 95.5% vs 9.1%** end-to-end (110 cases, 11 tasks; numeric 97.0/7.1, formula 98.0/10.1, faithful 100/43.4, missing 81.8/63.6, edge 100/80, false refusals 1.0/63.6; latency 60 s vs 112 s); **real-world 96.8% (30/31) vs 12.9% (4/31)** (numeric 96.3/3.7, faithful 100/25.9, false refusals 3.7/51.9). Base passes in real-world are only the 4 period-guard cases. Fine-tuned per task: 100% except ebitda 80%, gross_margin / multi_period / payback 90%.
- vs v1: on v1's 6 synthetic tasks 57/60 (v1 54/60); on rw-01..rw-21 21/21 (v1 16/21 raw); lakh rl-01..05 5/5. v2 wins.
- `configs/train.yaml`: `adapter_dir: qwen3_4b_finance_lora_v2` (default for run_assistant / eval_model / export; checked: contract resolves to v2). v1 adapter dir unchanged.
- `train_qwen3.py`: new `--overwrite`; refuses at startup when `adapter_dir` already holds `adapter_config.json` (before, a smoke test or a run with `train.yaml` would silently replace the evaluated v2 adapter at save time). Verified: plain `train_qwen3.py` now exits with the message; 101 fast tests pass.
- `MODEL_CARD.md`: v2 evaluation section (table, comparison with v1, failure analysis incl. rw-25); header / training-data line / intended use updated; only "GGUF vs adapter" left unmeasured.
- README: default adapter + overwrite guard note; v2 report path.
- Build progress 85% -> 87% (milestone 6: 11 -> 13; GGUF-vs-adapter still open).
- ~14:20 started `export.py --gguf q4_k_m` (default adapter = v2) in the background, log `outputs/export_v2_log.txt` (ends with `export exit code N`). Expected to download the 16-bit Qwen3-4B base for the merge.

## 2026-10-23 09:52 IST - v2 eval resumed
- Since the pause the user committed `3d1b196` "evaluation fix" and `7feb25e` "v2 training".
- Started the remaining v2 eval (resume checklist step 1): `eval_model.py --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2 --models both`, log `outputs/eval_v2_log.txt` after a `RESUMED` marker; cached 137 fine-tuned + 5 base rows are reused (4 fine-tuned real-world + 136 base cases left, ~4.5-5 h). The command appends `eval_model exit code N` to the log for the watchdog `outputs/watch_eval_v2.sh`.
- RPCS3 was running on the GPU at start (1.7 GB VRAM, 28% util); generation ran at 10.7-12.8 tok/s (normal 14-16), no spill.
- 09:59 fine-tuned v2 reruns: rw-20 OK, rw-21 OK, rw-26 OK, **rw-25 FAIL**: preprocessing worked ("Q3 FY26 (Oct-25, Nov-25, Dec-25)") and the model quoted all 3 correct values, but it framed the question as "YTD" with "FY26 starts in Oct-25", invented a need for last year's figures and refused (no tool call). Model error, not scorer. Likely data gap: no training question names a quarter as an explicit month list or asks "how many ... in <quarter>" from monthly rows. Candidate fix for v3 data: a `range_total` phrasing "in Q3 FY26 (Oct-25, Nov-25, Dec-25)".
- 10:00 interim re-score of the fine-tuned v2 run (`--models ft --rescore`, report in the scratchpad only; the 7 end-to-end changes are the known scorer fixes: 6 burn-multiple closing-opening cases + rw-24): **synthetic 95.5% end-to-end** (110 cases, 11 tasks; numeric 97.0%, formula 98.0%, faithful 100%, missing 81.8%, false refusals 1.0%) and **real-world 96.8% (30/31)** (only rw-25 fails; retrieval recall 100%). v1 for comparison: synthetic 90.0% (6 tasks), real-world 76.2% raw / 20 of 21 with the guard. On v1's 6 synthetic tasks v2 scores 57/60 (95%). Fine-tuned failures: raft-1020001 (adjusted EBITDA margin values), raft-1020006 + raft-1090006 (called the calculator on a missing-data case), raft-1050007 (payback false refusal), raft-1100008 (4-month range total).
- Gotcha: `TZ=Asia/Kolkata date` in Git Bash silently prints UTC; use PowerShell `Get-Date` for IST.

## 2026-10-22 21:02 IST - Session paused for the day (state saved for tomorrow)
- User asked to stop and continue tomorrow. Stopped: the v2 eval (pid 15048, base model at 5/141), the queued chain scripts (`run_eval_v2_after_train.sh`, `finish_eval_v2.sh`), watchdogs and keep-awake helpers. No GPU job running.
- v2 eval cache: fine-tuned 141/141 done; removed rw-20, rw-21, rw-25, rw-26 from `outputs/eval_runs/fine_tuned_v2.jsonl` (ran with the old period preprocessing) -> 137 rows; full copy kept as `fine_tuned_v2.before_period_fix.jsonl`. `base_v2.jsonl` 5 rows. No time-limit hits. All rows parse.
- `outputs/finish_eval_v2.sh` fixed before stopping (waits for the wrapper's exit line; keep-awake uses the Windows pid via `/proc/$$/winpid`); not needed tomorrow because the stale rows are already removed.
- Resume checklist written to memory.md section 1 "Next steps": one eval command (cached cases skipped), then `--rescore`, compare, model card, switch default adapter, GGUF export + server eval.
- Build progress unchanged today since 18:30: 85%.

## 2026-10-22 20:59 IST - Period preprocessing fixes (relative phrases, grammar, quarter months)
- v2 fine-tuned real-world run finished (old scorer: 127/141 overall). rl-01..rl-04 (lakh workbook) all OK incl. both from_lakh conversions (rl-03 LTV:CAC, rl-04 payback); rw-12 (v1's off-by-one column) now OK; rw-16..19 answered by the period guard.
- rw-25 ("new customers in Q3 FY26") failed: the model mapped Q3 FY26 to Jul-Sep 25 (Indian FY Q3 = Oct-Dec) and called it YTD. Quarter -> month mapping is a calendar fact, so it moves to deterministic preprocessing like "last month".
- Bug found while checking: `resolve_relative_periods` rewrote "previous month" / "the month before" to the data's second-latest month even when the question named a month. rw-26 reached the model as "in Feb-26 vs Feb-26?" (v2 still answered right from context); training-style MoM questions ("grow in Mar-26 versus the previous month?") would be corrupted the same way. It also left broken grammar ("What was CM2 Mar-26?", "the Feb-26").
- `common/periods.py` `resolve_relative_periods`: month phrases are rewritten only when the question has no explicit month (quarter phrases: no explicit quarter); a leading "the" is part of the match and "in" is inserted when no preposition precedes (`_PREPOSITIONS`): "net revenue last month" -> "net revenue in Mar-26"; a quarter with no column in the data but all 3 months present is spelled out: "Q3 FY26" -> "Q3 FY26 (Oct-25, Nov-25, Dec-25)" with a note. Quarters that have a column (quarterly tables) or whose months are missing are untouched.
- `tests/test_periods.py`: expectations updated to the grammatical form; new tests for explicit-month questions and quarter spelling. 101 fast tests pass; real-world retrieval recall 100% (31).
- Not changed: v1 compat cases rw-20/rw-21 now get "in Mar-26" / "in Feb-26" wording (only matters if the v1 run is ever regenerated; re-scoring is unaffected).
- Plan: after the v2 base run, delete rw-20, rw-21, rw-25, rw-26 from `outputs/eval_runs/fine_tuned_v2.jsonl` and `base_v2.jsonl`, rerun the same eval command (only those 4 x 2 run), then `--rescore` for the final v2 report.

## 2026-10-22 20:46 IST - Scorer allows month counts 1-12
- v2 eval rw-24 (YTD gross revenue FY26 to Dec-25): correct values, failed only because the report row says "Sum of 9 months" and 9 was not an allowed constant. The training template writes the month count in the report, so any YTD with 7, 9, 10 or 11 months would fail even for a perfect answer (the 30-per-task gold self-test had not drawn one).
- `common/scoring.py` `ALLOWED_CONSTANTS` = {0, 0.5, 0.75, 14, 15, 18, 24, 40, 100} + integers 1-12.
- Checks: gold self-test 100% on 200 multi-period cases and on all 11 tasks (30 each); 99 fast tests; v1 re-score: 0 changes.
- v2 eval (old scorer, in process): fine-tuned 118/130 at 20:40; real-world rw-01..rw-11 all OK.

## 2026-10-22 20:00 IST - Scorer: optional helper intermediates; eval_model --rescore
- v2 eval, fine-tuned 72/81 at ~20:00. 6 of the 9 failures were burn multiple cases where only `gross_new_arr` was "missing": the v2 model computed net new ARR as closing ARR - opening ARR (an equally valid route; net new ARR 220,210 and burn multiple 2.19x correct in raft-1070000), so the bridge-route intermediate never appeared. Scorer false negatives, not model errors.
- `common/scoring.py`: `Gold.optional` + `optional_from_steps(steps)`: helper-formula steps (total / difference / ratio / percent_of / margin_pct) whose result feeds only other helper steps. A miss on such a key is dropped from the checks; everything else is unchanged. In the v2 cases this covers `gross_new_arr` (burn multiple) and `revenue_leakage` (net revenue; leakage % stays required). `eval_model.synthetic_cases` sets it.
- `eval_model.py --rescore`: no LLM; rebuilds the cases for the given flags, re-scores the cached run files with the current scorer, prints every end-to-end change, writes the report.
- Checks: v1 run re-scored (`--rescore --v1-compat --tag v1`): 0 changes (v1 numbers stand). Scorer self-test 100% on all 11 tasks. Test `test_optional_intermediates` (closing-opening route passes, wrong answer still fails). 99 fast tests pass.
- The running v2 eval process loaded the old scorer: after it finishes, run `eval_model.py --rescore --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2 --models both` for the final report.

## 2026-10-22 18:30 IST - v2 adapter trained; v2 eval running
- v2 training finished 17:19 (exit 0; harmless resource_tracker traceback): 338/338 steps, 1 epoch, train runtime 326.5 min (incl. the 66-min standby pause), peak VRAM 6.19 GB. Loss 1.253 (step 5) -> 0.38 (20) -> 0.089 (40) -> 0.018 (100) -> ~0.012-0.014 (150-275) -> 0.0098 (300) -> 0.0095-0.0118 (320-335). Same shape as v1, ending at the same level (~0.01).
- Saved `qwen3_4b_finance_lora_v2/` (adapter_model.safetensors 264 MB, tokenizer, chat_template.jinja, loss_history.json, loss_curve.png, `contract.json` = {"contract": "v2", "prompt_version": "5c6b089a25"}). Checkpoints in `outputs/v2/`.
- The laptop went into Modern Standby again between the end of training (keep-awake helper ended with the training pid) and 18:29 (Kernel-Power 507 exit at 18:29:36), so the queued eval started at 18:29 instead of ~17:20. The eval process blocks sleep itself (boost_process: "sleep blocked while running"). Lesson: a chain script must hold the keep-awake across the hand-off.
- `eval_model.py --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2 --models both` running since 18:29 (log `outputs/eval_v2_log.txt`); watchdog `outputs/watch_eval_v2.sh` (git-ignored).

## 2026-10-22 16:13 IST - v2 eval queued behind training
- v2 training progress: step 250/338, loss 0.0121 (step 50: 0.057, 75: 0.035, 100: 0.018, 125: 0.014, 150: 0.014, 175: 0.017, 200: 0.012, 225: 0.014). `outputs/v2/checkpoint-50 ...` saved without stalls. ETA ~17:30.
- New `outputs/run_eval_v2_after_train.sh 8436` (git-ignored), started in the background: waits for the training to exit; only if the log says exit code 0 and `qwen3_4b_finance_lora_v2/` has `adapter_config.json` + `contract.json`, runs `eval_model.py --config configs/eval_v2.yaml --adapter qwen3_4b_finance_lora_v2 --tag v2 --models both` (fine-tuned then base; 110 synthetic + 31 real-world; est. ~3 h + ~4.5 h). Log `outputs/eval_v2_log.txt`. The eval process blocks sleep itself (`boost_process`).
- Watchdog now reports loss every 25 steps only.

## 2026-10-22 13:18 IST - v2 training paused 66 min by Modern Standby; keep-awake fix
- Step 21 of the v2 run took ~67 min (12:08-13:15). Windows event log: Kernel-Power 506 "entering Modern Standby" 12:08:20, 507 "exiting" 13:14:41. The laptop slept; training resumed by itself at normal speed (steps 22-23 ~40 s). Not a VRAM spill. Loss so far: 1.253 (step 5), 0.799 (10), 0.51 (15), 0.3825 (20).
- Current run: `outputs/keep_awake.ps1 -ProcessId 8436` (git-ignored) started in the background at 13:17: holds `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` until the training exits; no power settings changed. Closing the lid can still sleep.
- Permanent: `common/winperf.py` `boost_process()` now also blocks idle sleep for the life of the process (same API; prints "sleep blocked while running"). Applies to run_assistant / eval_model / train_qwen3 / export.
- Watchdog `outputs/watch_train_v2.sh` rewritten: tqdm's s/it is a smoothed average (one 67-min step kept it at 600+ s/it for several steps, a false "slow" alarm), so it now alerts only when the step counter has not moved for 10 min; prints only new loss values.
- New ETA: ~17:20 IST (315 steps x ~46 s from 13:16).

## 2026-10-22 11:57 IST - v2 training running normally
- `train_qwen3.py --config configs/train_v2.yaml` (pid 8436): 2,700 train / 300 test examples, 2 dropped (> 6,144 tokens), masking check 576/3,264 tokens of example 0 carry loss, 338 steps, 45-47 s/step -> ETA ~4 h 15 min (about 16:15 IST). VRAM 6.0 GB used, GPU 100%.
- The training process shows 1.8 GB of Windows "shared GPU memory", but step time equals healthy v1 training (~46 s), so it is not on the hot path. Watchdog `outputs/watch_train_v2.sh` (git-ignored) reports loss values, flags steps slower than 120 s (spill signature), errors and the finish.

## 2026-10-22 11:52 IST - v1 evaluation complete; model card filled; v2 training started
- `eval_model.py --models both --n-per-task 10 --tag v1 --v1-compat` finished 11:51 (exit 0; the traceback at exit is the known harmless multiprocess resource_tracker message). No time-limit hits after the compat relaunch. Report `reports/eval_2026-10-22_v1.md`.
- Synthetic (60 cases, oracle context), fine-tuned vs base: end-to-end 90.0% vs 15.0%; numeric 90.7% vs 9.3%; formula 96.3% vs 9.3%; faithfulness 100% vs 40.7%; missing-data 83.3% vs 66.7%; edge cases 100% vs 66.7%; false refusals 3.7% vs 59.3%; latency 147 s vs 139 s per case. Fine-tuned by task: break_even, contribution_margin, net_revenue 100%; payback 90%; ebitda 80%; ltv_cac 70%.
- Real-world (21, RAG, raw model, no guard): end-to-end 76.2% vs 9.5%; numeric 94.1% vs 0%; faithfulness 100% vs 47.1%; missing-period 0/4 vs 2/4. Fine-tuned failures: rw-12 (column off by one) + rw-16..19 (absent month). With the deterministic period guard (fires on exactly rw-16..19) the pipeline would be 20/21; not measured as a separate run.
- Base model: mostly no calculator call (false refusals 59-65%), invented numbers.
- `MODEL_CARD.md`: evaluation section filled with these numbers and the failure analysis.
- 11:51 v2 training started automatically (`outputs/run_v2_after_eval.sh`), log `outputs/train_v2_log.txt`.

## 2026-10-22 11:24 IST - v2 training queued behind the v1 eval
- New `outputs/run_v2_after_eval.sh <pid>` (git-ignored): waits until the v1 eval process (pid 26968) exits, then runs `train_qwen3.py --config configs/train_v2.yaml`; log `outputs/train_v2_log.txt` (start line, trainer output, exit code). Started in the background by Claude at 11:24. No separate 5-step smoke test: the full run fails just as early on setup errors.
- Expected: ~375 steps (3,000 examples, 2 dropped as too long, 1 epoch, effective batch 8) at ~46 s/step = ~4.5-5 h; adapter -> `qwen3_4b_finance_lora_v2/` with `contract.json`.
- v1 eval at 11:24: base 65/81 cached, ~16 cases left (~25 min).

## 2026-10-22 11:20 IST - QoQ variant; v2 dataset regenerated
- `generate_raft_data.py`: `QOQ_METRICS` (net revenue, S&M spend, ending customers) and `_plan_qoq`: quarter-on-quarter change + `growth_pct` read directly from the quarterly summary (current quarter = latest 70%, else an earlier one). `plan_multi_period` uses it 20% of the time when >= 2 fiscal quarters exist. Variant `qoq:<key>`; quarter periods, so no period-gap examples; ending customers is a count (no lakh note).
- `dataset_v2.jsonl` regenerated: 3,000 examples, 0 mismatches; multi_period variants in the 3,000 trial: mom 153, range_total 39, qoq 37, average 33, ytd 24. Tokens median 3,511, p90 4,773, max 6,204 (2 over 6,144).
- `tests/test_multi_period.py`: qoq branch; sample raised to 300 examples so every variant appears. 98 fast tests pass; scorer self-test 100% on all 11 tasks.
- `MODEL_CARD.md`: v2 training-data capabilities, units and prompt-contract limitations, retrieval recall 31/31 (eval numbers still pending).

## 2026-10-22 11:17 IST - Messy Excel layouts: stacked tables, numbers as text, transposed tables, currency titles
- Probe workbook (scratchpad `messy_mis.py`) showed 5 failures in the old Excel ingester: text numbers passed raw ("1,23,456.50", "(12,345.00)" - a parenthesised negative could reach the calculator as positive); a second table on the same sheet was not detected (its header became a data row); dates inside data cells printed as "2026-01-01 00:00:00"; a months-down-the-rows table had no periods; "(INR)" in a title gave no units.
- `ingest_financials.py` Excel rewrite: `_excel_blocks` (every row with >= 2 period cells is a table header; label-only rows right above a later header, after a blank line, are its title rows; no period header -> old single-table fallback `_header_row`); `_is_period`; `parse_number_text` (`_NUM_TEXT_RE`: Indian 12,34,567 and western 1,234,567 grouping, decimals, (parentheses) / leading minus negatives, optional ₹ / Rs. / INR / $ / USD; "12,34" rejected); `_cell_text` (dates -> Mon-YY, number text -> plain number); months-down-the-rows tables transposed to `Line Item | Jan-26 | ...`; columns with no header that are empty within a table are dropped; later tables inherit a sheet-wide units note unless their title reads like KPIs (`NON_MONEY_TITLE_RE`); plain currency fallback; duplicate chunk ids get a `::k` suffix.
- `common/units.py`: `detect_currency` / `first_currency` ("Cash Summary (INR)", "Amounts in USD", "Currency: INR").
- Regression guard: ingestion output for the sample workbook and the lakh workbook is byte-identical to before the rewrite (all 22 chunks: ids, text, metadata).
- Bash heredoc gotcha hit again: `\b` in the new `_CUR_ONLY` regex became backspace bytes; fixed with a byte-level script (scratchpad `fix_backspace.py`). Also repaired two pre-existing stray backspace bytes in the docs (Changelog 2026-10-21 16:55 entry, memory.md section 3 grep hint). Repo scan for control bytes is clean.
- Tests: `tests/test_messy_excel.py` (9 number-text cases, currency detection, the messy workbook end to end). 98 fast + 1 slow pass. Real-world retrieval recall 100% (31). Compat re-score unchanged.

## 2026-10-22 11:06 IST - v2 training prepared (config, dataset, eval config); --adapter; export records the contract
- `configs/train_v2.yaml` (new): `dataset_v2.jsonl`, `output_dir: outputs/v2`, `adapter_dir: qwen3_4b_finance_lora_v2`, 1 epoch, `eval: false`; everything else as v1 (r32, lr 2e-4, bs 1 x ga 8, max_seq_length 6144). `configs/train.yaml` unchanged (still the v1 adapter, which eval/run_assistant/export use by default).
- `dataset_v2.jsonl` generated (`generate_raft_data.py --n 3000 --output dataset_v2.jsonl`): 3,000 examples, 11 tasks, contract v2, 0 mismatches; tokens median 3,503, p90 4,773, max 6,204 (2 over 6,144 -> dropped by training). `dataset.jsonl` (v1) kept.
- `configs/eval_v2.yaml` (new): `tasks: null` (all 11), `n_per_task: 10`.
- `eval_model.py`: `--adapter DIR` (default train.yaml adapter_dir); the contract default and report header use it.
- `export.py`: writes `exports/contract.json` ({contract, prompt_version, adapter}); SERVING.md tells the user to pass `--contract <c>` in server mode.
- `.gitignore`: `*_lora_v*/` (the v2 adapter dir was not covered by `*_lora/`).
- README: v2 data / train / eval commands.

## 2026-10-22 11:04 IST - Lakh-format sample MIS + 5 real-world lakh cases
- `make_sample_mis.py`: `LAKH`, `LAKH_KEYS` (all P&L rows + cash rows), `build_data_lakh()` (those rows / 100,000, 2 decimals; net income and the cash roll-forward recomputed on the rounded numbers), `build_workbook(out, lakh=False)`, CLI `--lakh`. Lakh workbook: P&L units row "INR (₹ in lakhs)", cash sheet units only in its title "Cash Position (Rs. in Lakhs)", money format `#,##0.00`, unit economics in INR WITHOUT the "Total S&M Spend" row (so CAC needs the lakh marketing rows + `from_lakh`).
- `eval_cases.py`: `build_lakh_cases()` rl-01 net revenue Mar-26, rl-02 EBITDA Feb-26, rl-03 LTV:CAC Jan-26 (requires from_lakh; CAC ~INR per customer), rl-04 payback Feb-26 (requires from_lakh), rl-05 average payroll Jan-26..Mar-26 (lakh). All `edge_cases` include `scaled_units`.
- `eval_model.py`: `realworld_cases` builds `outputs/sample_startup_mis_lakh.xlsx` fresh each run and evaluates groups (main workbook, lakh workbook) with one retriever re-indexed per file (`_realworld_group`). `--v1-compat` skips the lakh group (and rw-22+).
- Checks: retrieval recall 100% on all 31 real-world cases; ingested units: Revenue / Bank "INR lakh" (units-row normalisation and title fallback both work), Customer Metrics "INR unless stated"; compat re-score unchanged; scorer self-test 100%.
- Tests: `test_lakh_sample_workbook_units_and_cases` (units per section, no S&M row, identities, finite truths, CAC magnitude). 87 fast tests pass.
- Eval progress: base 55/81 (21 of 47 reruns done at 11:04), no time-limit hits since the compat relaunch.
- README: real-world suite description (31 cases, lakh workbook).

## 2026-10-22 10:52 IST - Multi-period task family; 5 real-world multi-period cases; scorer thresholds fixed
- `common/formulas.py`: `growth_pct(current, prior)` = (current - prior) / |prior| (error if prior is 0), `average(values)` (error if empty). `common/tools.py`: registered (`growth_pct` percent; `average` currency, list input). Contract v2 only: v1 tool list unchanged (still `2013a079fb`); v2 prompt version now `5c6b089a25`.
- `generate_raft_data.py`: task `multi_period` (`plan_multi_period`, `MP_METRICS`, `_mp_table`, `fy_start_index`). Metrics: gross revenue, COGS, performance marketing, payroll, new customers (flows) + closing ARR, closing cash, active customers (stocks, MoM only). Variants: `mom` (difference + growth_pct vs the previous month), `range_total` (total over the 3-6 months ending P), `average` (total + average), `ytd` (Indian FY from April to P). Variant string `<variant>:<metric>`. Questions name only the end period, so period-gap examples work. `Plan.money_in_stmt_units` (False for count metrics: no lakh note / header). `STATEMENT_TASKS` += multi_period.
- `configs/data.yaml`: `task_weights.multi_period: 1.25`.
- `eval_cases.py`: rw-22 (gross revenue growth Mar-26 vs Feb-26), rw-23 (average monthly payroll Jan-26 to Mar-26), rw-24 (YTD gross revenue FY26 up to Dec-25), rw-25 (new customers in Q3 FY26 = Oct-Dec total), rw-26 (active customers % change Feb-26 vs the month before). Worded unlike the training templates. `V1_CASE_IDS` (rw-01..rw-21); `eval_model.realworld_cases(..., v1_compat)` keeps only those under `--v1-compat`. Retrieval recall 100% on all 26; the period guard does not fire on the new ones.
- `common/scoring.py`: `ALLOWED_CONSTANTS` += 0.5, 0.75, 6, 18, 40 (template thresholds). The gold self-test had only ever run on the 6 v1 tasks; over all 11 tasks it flagged correct magic-number / runway reports (0.75 / 0.5 / 6 in the CFO take) as unfaithful (95.2% end-to-end). Now 100% on 330 gold cases (30 per task). Re-scoring the stored v1 rows: no pass/fail change (one base row's hallucinated-number list shrinks).
- Trial: 3,000 examples incl. 286 multi_period (mom 169, average 48, range_total 46, ytd 23; 25 missing), 0 mismatches.
- Tests: `tests/test_multi_period.py` (formulas; 120 examples validate; each variant uses the right formulas; YTD starts in April). 86 fast tests pass.
- README: multi-period task list; real-world suite is 26 cases.

## 2026-10-22 10:43 IST - v1 eval made reproducible; v2 data: phrasing, quarterly note, lakh/crore units; prompt contracts
**Evaluation (v1)**
- Last night's base run stopped at 46/81 (~00:58; laptop off). Slowdowns came back after the manual 22:55 pin (many turns ~6 tok/s; affinity alone was not enough, power throttling was still on). 12 base cases had hit the 240 s limit: raft-1000004, 1010001, 1010002, 1020004, 1020005, 1020006, 1030002, 1030003, 1030005, 1030006, 1040004, 1040005. Removed them from `outputs/eval_runs/base_v1.jsonl` (backup `base_v1.backup_2026-10-22.jsonl`) so they rerun; greedy decoding makes all other rows speed-independent.
- 10:14 resume was stopped after ~5 min (no rows written): the restart loaded the post-22:55 code, so the base model would have been scored on different inputs than the fine-tuned model (row-decimal rendering, missing cases turned into period-gap cases, period guard on for real-world).
- New `eval_model.py --v1-compat`: `common.tables.ALIGN_ROW_DECIMALS = False`, data cfg `compat_v1: true` (no period gaps / question variation / quarterly note / scaled units), period guard off, prompt contract v1. Verified by rebuilding the cases and re-scoring every stored trace: 81/81 fine-tuned (incl. 21 real-world) and all cached base rows reproduce exactly (0 differences); without the flag 2 missing cases change.
- 10:22 relaunched `eval_model.py --models both --n-per-task 10 --tag v1 --v1-compat` (log `outputs/eval_v1_log.txt`). CPU boost active from the start: steady 15.9-16.1 tok/s. raft-1020005 now passes (was a time-limit FAIL).

**Prompt contracts** (so the v1 adapter keeps seeing exactly what it was trained on)
- `common/tools.py`: `V1_FORMULAS` (the 25 original formulas); `formula_signatures(names)`.
- `common/prompts.py`: `CONTRACT = "v2"`, `tool_description(contract)`, `build_tools(contract)`, `prompt_version(contract)`, `adapter_contract(adapter_dir)` (reads `<adapter>/contract.json`; missing file = v1; no adapter = current). v1 hash reproduces `2013a079fb` (the v1 dataset's version; pinned by a test). v2 = `553efd71f4`; v2 tool description adds a units sentence + 4 conversion formulas.
- `run_assistant.py`: `FinanceLLM(..., contract=None)` uses the adapter's contract; `ServerLLM(..., contract=)`; `--contract v1|v2` (default: the configured adapter's). `eval_model.py`: `--contract`; base model gets the same contract as the adapter it is compared with; report header shows the contract.
- `train_qwen3.py`: writes `<adapter>/contract.json` (`contract`, `prompt_version`) after training.
- `generate_raft_data.py`: `PROMPT_VERSION = prompt_version()` (v2). The on-disk v1 `dataset.jsonl` is now stale for training (expected; regenerate for v2).

**Calculator**: `common/formulas.py` `UNIT_FACTORS`, `to_base_units`, `from_lakh`, `from_crore`, `from_thousands`, `from_millions`; registered in `REGISTRY` (currency).

**Generator (v2 data)**
- `vary_question` + `EXTRA_QUESTIONS` (3-5 extra founder phrasings per task except break-even), `Q_PREFIXES` / `Q_SUFFIXES`, casual lower-casing, dropped "?"; own rng stream (`question|seed|idx`). 3,000 trial: ~1,800 distinct question forms.
- Quarterly-summary distractor note in `<think>` for month tasks ("[k] has quarter totals ...; I need <month> monthly figures") - targets the v1 LTV:CAC failure (quarterly S&M used as monthly). In 539/2,723 tool examples.
- Lakh / crore statements: `configs/data.yaml` `scaled_units: 0.2`; own rng stream (`units|seed|idx`). `scale_company` (P&L, cash, ARR, quarterly, working capital, one-offs in INR lakh / INR crore (if gross revenue >= 3 crore in every window month, 50%) / USD thousands; 2 decimals; derived rows recomputed so identities hold); `Company.money_unit/money_units/money_factor/stmt_units`; `NON_MONEY_KEYS` (counts, ARPU, %, unit-economics S&M stay plain). LTV:CAC / payback with S&M summed from a scaled P&L: `total` -> `from_<unit>` -> `cac` (sm given probability 0.25 for scaled companies so the conversion case is common: 88 per 3,000). Statement tasks report `Currency: INR lakh` and get a units note; P&L-derived gross margin notes that units cancel.
- Inputs quoted in `<think>` now use the exact rendered cell (`Table.cell`, incl. row-decimal alignment): "3.00", not "3".
- Bug fixed (pre-existing): a missing-data example that deleted a whole row (magic number: both quarters of Net Revenue on one row) still quoted the other quarter's value from the deleted row. `apply_missing` now blanks only the cell when another input shares the row.
- `validate_example`: every quoted input must be a cell of that row in the context.
- Trial: 3,000 examples, 0 mismatches; 620 scaled-unit companies (302 USD thousands, 262 INR lakh, 52 INR crore); 14,629 quoted inputs all verbatim.

**Ingestion**
- New `common/units.py`: `detect_units` ("₹ in lakhs", "Rs. Lakh", "INR Cr", "USD '000", "$000s", "($ in millions)", "All figures in crores" -> "INR lakh" / "INR crore" / "USD thousands" / "USD millions"; never bare k/m/cr without a currency; lakh/crore only with INR), `first_units`.
- `ingest_financials.py`: Excel normalises `Units:` rows and falls back to title rows / header cells / sheet name; PDF tables get units from caption/headers/section, else the last 3 text items of the section, else a document-wide units statement (`UNITS_CUE_RE`: only text that reads like a units statement). PDFs previously had no units at all.

**Tests**: `tests/test_units_contract.py` (new: formulas, calculator conversion, frozen v1 contract, adapter_contract, scaled examples for every task, misquoted-input detection, 16 unit phrasings, Excel title units); `tests/test_dataset.py` (+question variation, compat has no period gaps, quarterly note). 84 fast + 1 slow pass. Scorer self-test 100%. Real-world retrieval recall still 100%.

**Docs**: README (eval `--v1-compat`, units in data and ingestion, prompt contracts, units.py in layout); memory.md sections 1, 3-8.

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
- Bug: the heredoc patch wrote literal backspace bytes instead of `\b` in those regexes; fixed with an exact edit; repo scanned, no other control characters.
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
