"""Export the fine-tuned model for serving.

  merged  -> exports/merged_16bit/   full 16-bit weights (vLLM, transformers). ~8 GB.
  gguf    -> exports/gguf/           quantized GGUF (llama.cpp llama-server, Ollama, LM Studio). q4_k_m ~2.5 GB.

Also writes exports/gguf/Modelfile (Ollama) and exports/SERVING.md with serve commands.
GGUF conversion uses Unsloth's llama.cpp (on this machine a prebuilt CUDA build lives in ~/.unsloth/llama.cpp).

Usage:
  python export.py --gguf q4_k_m            # default
  python export.py --gguf q4_k_m q8_0 --merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

SERVING = """# Serving the exported model

The assistant needs a server that speaks the OpenAI chat API with tool calling. Then run:

    python run_assistant.py --backend server --server-url http://localhost:8080/v1 --server-model {name}

## llama.cpp (Windows/Linux/macOS, GPU or CPU)

    llama-server -m exports/gguf/{gguf} --jinja -ngl 99 -c 8192 --port 8080 --alias {name}

`--jinja` enables Qwen3's chat template with tool calls. Unsloth's prebuilt binary on this machine:
`%USERPROFILE%\\.unsloth\\llama.cpp\\build\\bin\\Release\\llama-server.exe`.

## Ollama

    cd exports/gguf
    ollama create {name} -f Modelfile
    ollama serve            # API at http://localhost:11434/v1
    python run_assistant.py --backend server --server-url http://localhost:11434/v1 --server-model {name}

## vLLM (Linux, merged 16-bit weights)

    vllm serve exports/merged_16bit --served-model-name {name} --max-model-len 8192 \\
        --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
    python run_assistant.py --backend server --server-url http://localhost:8000/v1 --server-model {name}
"""

MODELFILE = """FROM ./{gguf}
# Qwen3 thinking-mode sampling defaults
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_ctx 8192
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", nargs="*", default=["q4_k_m"], help="GGUF quantizations (empty list = skip)")
    ap.add_argument("--merged", action="store_true", help="also save merged 16-bit weights")
    ap.add_argument("--adapter", help="adapter dir (default from configs/train.yaml)")
    ap.add_argument("--out", default=str(ROOT / "exports"))
    ap.add_argument("--name", default="af-finance-qwen3-4b")
    args = ap.parse_args()
    from common.winperf import boost_process

    boost_process()  # keep the GPU-feeding thread off Windows efficiency cores

    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    adapter = Path(args.adapter) if args.adapter else ROOT / cfg["adapter_dir"]
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"No adapter at {adapter}. Run train_qwen3.py first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(adapter), max_seq_length=cfg["max_seq_length"],
                                                         load_in_4bit=True, dtype=None)
    if args.merged:
        target = out / "merged_16bit"
        model.save_pretrained_merged(str(target), tokenizer, save_method="merged_16bit")
        print(f"merged 16-bit -> {target}")

    gguf_name = "<model>.gguf"
    if args.gguf:
        target = out / "gguf"
        model.save_pretrained_gguf(str(target), tokenizer, quantization_method=args.gguf)
        files = sorted(target.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
        preferred = [f for f in files if args.gguf[0].lower() in f.name.lower()]
        if files:
            gguf_name = (preferred or files)[-1].name
        (target / "Modelfile").write_text(MODELFILE.format(gguf=gguf_name), encoding="utf-8")
        print(f"GGUF -> {target} ({', '.join(f.name for f in files)})")

    (out / "SERVING.md").write_text(SERVING.format(name=args.name, gguf=gguf_name), encoding="utf-8")
    print(f"serve instructions -> {out / 'SERVING.md'}")


if __name__ == "__main__":
    main()
