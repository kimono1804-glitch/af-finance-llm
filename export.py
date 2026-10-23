"""Export the fine-tuned model for serving.

  merged  -> exports/merged_16bit/   16-bit HF weights (vLLM, transformers, GGUF source). ~8 GB.
  gguf    -> exports/gguf/           GGUF files (llama.cpp llama-server, Ollama, LM Studio) + Ollama Modelfile.

Also writes exports/SERVING.md (serve commands) and exports/contract.json (prompt contract of the adapter).

Why a custom merge: QLoRA trains the adapter on the DEQUANTIZED 4-bit base, y = x @ deq(nf4(W)) + x @ BA. Unsloth's
save_pretrained_merged / save_pretrained_gguf add BA to the original bf16 weights instead, which puts the nf4 rounding
error (~9% mean relative difference on the quantized layers of this model) back under the adapter. On 2026-10-23 that
merge lost 14 points end-to-end (synthetic 95.5% -> 81.8% at q4_k_m; the same regression at q8_0). This script merges
into deq(nf4(W)), which reproduces the model the adapter was trained and evaluated with.

Quantization: q8_0 is the default and the verified type. Lower types re-round weights that already sit on the nf4
grid; q4_k_m from this merge was clearly worse in a 5-case check. Evaluate any other type before shipping it.

Usage:
  python export.py                       # merge + q8_0 GGUF + Modelfile + SERVING.md
  python export.py --gguf q8_0 q6_k      # several types (each needs its own eval)
  python export.py --gguf                # merge only (no GGUF)
  python export.py --finalize-only       # no merge/convert: rewrite Modelfile / SERVING.md / contract.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_LLAMA_CPP = Path.home() / ".unsloth" / "llama.cpp"  # needs convert_hf_to_gguf.py + a llama-quantize binary
QUANT_STATS = ("absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4")
SHARD_BYTES = 4_500_000_000

SERVING = """# Serving the exported model

The assistant needs a server that speaks the OpenAI chat API with tool calling. Then run:

    python run_assistant.py --backend server --server-url http://localhost:8080/v1 --server-model {name} --contract {contract}

This model was trained on prompt contract **{contract}** (`exports/contract.json`): the client must send that
contract's tool list, so always pass `--contract {contract}` to run_assistant.py / eval_model.py in server mode.

## llama.cpp (Windows/Linux/macOS)

    llama-server -m exports/gguf/{gguf} --jinja -ngl 99 -c 8192 -np 1 -fa on --port 8080 --alias {name}

`--jinja` enables Qwen3's chat template with tool calls. `-ngl 99` needs a GPU build of llama.cpp (CUDA / Vulkan /
Metal); a CPU-only build ignores it. On an 8 GB GPU a q8_0 model uses ~5.4 GB.

## Ollama

    cd exports/gguf
    ollama create {name} -f Modelfile
    ollama serve            # API at http://localhost:11434/v1
    python run_assistant.py --backend server --server-url http://localhost:11434/v1 --server-model {name} --contract {contract}

## vLLM (Linux, merged 16-bit weights)

    vllm serve exports/merged_16bit --served-model-name {name} --max-model-len 8192 \\
        --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
    python run_assistant.py --backend server --server-url http://localhost:8000/v1 --server-model {name} --contract {contract}
"""

# Qwen3 tool-calling template for Ollama (Ollama uses Go templates, not the GGUF's Jinja template).
OLLAMA_TEMPLATE = '''{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}'''

MODELFILE = '''FROM ./{gguf}
TEMPLATE """{template}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.6
PARAMETER min_p 0.0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1
PARAMETER num_ctx 8192
'''


def merge_tensor(w, lora_a=None, lora_b=None, scale: float = 1.0):
    """W' = W + scale * B @ A in float32, returned as bfloat16. W is the (dequantized) base weight."""
    import torch

    w = w.float()
    if lora_a is not None:
        w = w + scale * (lora_b.float() @ lora_a.float())
    return w.to(torch.bfloat16)


def merge_dequant(adapter: Path, base_model: str, out: Path, device: str = "cpu") -> None:
    """Merge the LoRA adapter into the dequantized nf4 base and save 16-bit HF shards + config + tokenizer to out."""
    import torch
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = Path(base_model) if Path(base_model).is_dir() else Path(
        snapshot_download(base_model, local_files_only=_cached(base_model)))
    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = acfg["lora_alpha"] / acfg["r"]
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    base_files = sorted(base_dir.glob("*.safetensors"))
    lora = safe_open(str(adapter / "adapter_model.safetensors"), "pt", device="cpu")
    lora_keys = set(lora.keys())
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shards: list[dict[str, torch.Tensor]] = []
    n_deq = n_lora = 0
    for bf in base_files:
        with safe_open(str(bf), "pt", device="cpu") as base:
            keys = set(base.keys())
            weights = sorted(k for k in keys if not any(k.endswith("." + s) for s in QUANT_STATS))
            for k in weights:
                t = base.get_tensor(k)
                if f"{k}.quant_state.bitsandbytes__nf4" in keys:
                    stats = {s: base.get_tensor(f"{k}.{s}").to(dev) for s in QUANT_STATS if f"{k}.{s}" in keys}
                    t = dequantize_4bit(t.to(dev), QuantState.from_dict(stats, device=dev))
                    n_deq += 1
                stem = k[: -len(".weight")] if k.endswith(".weight") else k
                a_key, b_key = f"base_model.model.{stem}.lora_A.weight", f"base_model.model.{stem}.lora_B.weight"
                if a_key in lora_keys:
                    t = merge_tensor(t.to(dev), lora.get_tensor(a_key).to(dev), lora.get_tensor(b_key).to(dev), scale)
                    n_lora += 1
                else:
                    t = t.to(torch.bfloat16)
                shard[k] = t.cpu().contiguous()
                shard_bytes += shard[k].numel() * 2
                if shard_bytes >= SHARD_BYTES:
                    shards.append(shard)
                    shard, shard_bytes = {}, 0
    if shard:
        shards.append(shard)
    expected = sum(1 for k in lora_keys if k.endswith("lora_A.weight"))
    if n_lora != expected:
        sys.exit(f"merged {n_lora} LoRA modules but the adapter has {expected}: tensor names do not match")

    weight_map = {}
    for i, s in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, str(out / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in s})
    total = sum(t.numel() * 2 for s in shards for t in s.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8")

    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if (base_dir / "generation_config.json").exists():
        shutil.copy(base_dir / "generation_config.json", out)
    # Tokenizer + chat template exactly as saved with the adapter (what training and eval used).
    for f in adapter.iterdir():
        if f.name.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges", "chat_template")):
            shutil.copy(f, out / f.name)
    print(f"merged (dequantized nf4 base) -> {out}: {n_deq} nf4 tensors dequantized, LoRA merged into {n_lora} "
          f"(scale {scale}), {len(shards)} shard(s)")


def _cached(repo: str) -> bool:
    """True if the HF repo is already in the local cache (then no network call is made)."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


def find_quantize(llama_cpp: Path) -> Path:
    for p in [llama_cpp / "build" / "bin" / "Release" / "llama-quantize.exe", llama_cpp / "build" / "bin" / "llama-quantize",
              llama_cpp / "llama-quantize.exe", llama_cpp / "llama-quantize"]:
        if p.exists():
            return p
    found = shutil.which("llama-quantize")
    if found:
        return Path(found)
    sys.exit(f"llama-quantize not found under {llama_cpp} or on PATH (set --llama-cpp)")


def to_gguf(merged: Path, target: Path, quants: list[str], llama_cpp: Path, stem: str, keep_bf16: bool) -> list[Path]:
    """HF shards -> bf16 GGUF (convert_hf_to_gguf.py) -> one file per quant type (llama-quantize)."""
    converter = llama_cpp / "convert_hf_to_gguf.py"
    if not converter.exists():
        sys.exit(f"{converter} not found (set --llama-cpp to a llama.cpp checkout)")
    target.mkdir(parents=True, exist_ok=True)
    bf16 = target / f"{stem}.BF16.gguf"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(converter), str(merged), "--outtype", "bf16", "--outfile", str(bf16)],
                   check=True, env=env)
    quantize = find_quantize(llama_cpp)
    files = []
    for q in quants:
        if q.lower() == "bf16":
            files.append(bf16)
            continue
        f = target / f"{stem}.{q.upper()}.gguf"
        subprocess.run([str(quantize), str(bf16), str(f), q.lower()], check=True)
        files.append(f)
    if not keep_bf16 and bf16 not in files:
        bf16.unlink()
    return files


def write_modelfile(target: Path, gguf_name: str) -> None:
    (target / "Modelfile").write_text(MODELFILE.format(gguf=gguf_name, template=OLLAMA_TEMPLATE), encoding="utf-8")


def write_serving(out: Path, adapter: Path, name: str, gguf_name: str) -> None:
    from common.prompts import adapter_contract, prompt_version

    contract = adapter_contract(adapter)
    (out / "contract.json").write_text(json.dumps({"contract": contract, "prompt_version": prompt_version(contract),
                                                   "adapter": adapter.name}), encoding="utf-8")
    (out / "SERVING.md").write_text(SERVING.format(name=name, gguf=gguf_name, contract=contract), encoding="utf-8")
    print(f"serve instructions -> {out / 'SERVING.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", nargs="*", default=["q8_0"], help="GGUF types (q8_0 verified; empty list = merge only)")
    ap.add_argument("--adapter", help="adapter dir (default from configs/train.yaml)")
    ap.add_argument("--out", default=str(ROOT / "exports"))
    ap.add_argument("--name", default="af-finance-qwen3-4b")
    ap.add_argument("--llama-cpp", default=str(DEFAULT_LLAMA_CPP), help="llama.cpp dir (convert_hf_to_gguf.py, llama-quantize)")
    ap.add_argument("--device", default="cpu", help="device for dequantization (cpu needs ~10 GB RAM, ~3 min)")
    ap.add_argument("--keep-bf16", action="store_true", help="keep the intermediate bf16 GGUF (8 GB)")
    ap.add_argument("--finalize-only", action="store_true",
                    help="no merge/convert: rewrite Modelfile (for the newest .gguf), SERVING.md and contract.json")
    args = ap.parse_args()
    from common.winperf import boost_process

    boost_process()  # performance cores, no power throttling, no idle sleep while exporting

    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    adapter = Path(args.adapter) if args.adapter else ROOT / cfg["adapter_dir"]
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"No adapter at {adapter}. Run train_qwen3.py first.")
    base_model = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    out = Path(args.out)
    merged, target = out / "merged_16bit", out / "gguf"
    stem = args.name

    if not args.finalize_only:
        merge_dequant(adapter, base_model, merged, device=args.device)
        if args.gguf:
            files = to_gguf(merged, target, args.gguf, Path(args.llama_cpp), stem, args.keep_bf16)
            print(f"GGUF -> {target} ({', '.join(f.name for f in files)})")

    ggufs = sorted(target.glob("*.gguf"), key=lambda p: p.stat().st_mtime) if target.is_dir() else []
    preferred = [f for f in ggufs if args.gguf and args.gguf[0].lower() in f.name.lower()]
    gguf_name = (preferred or ggufs)[-1].name if ggufs else "<model>.gguf"
    if ggufs:
        write_modelfile(target, gguf_name)
    write_serving(out, adapter, args.name, gguf_name)


if __name__ == "__main__":
    main()
