"""export.py: the LoRA is merged into the DEQUANTIZED nf4 base (what QLoRA trained on), not the original weights."""

import json

import pytest

torch = pytest.importorskip("torch")
bnb_f = pytest.importorskip("bitsandbytes.functional")
st = pytest.importorskip("safetensors.torch")

import export  # noqa: E402


def _fake_checkpoint(tmp_path):
    """A 1-layer bnb-4bit 'base' (one nf4 weight, one 16-bit norm) and a LoRA adapter for it."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 128, generator=g, dtype=torch.float32).to(torch.bfloat16)
    packed, qs = bnb_f.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    base = tmp_path / "base"
    base.mkdir()
    tensors = {"model.layers.0.mlp.up_proj.weight": packed,
               "model.norm.weight": torch.ones(128, dtype=torch.bfloat16)}
    for k, v in qs.as_dict(packed=True).items():
        tensors[f"model.layers.0.mlp.up_proj.weight.{k}"] = v
    st.save_file(tensors, str(base / "model.safetensors"))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization_config": {"load_in_4bit": True}}))

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    a = torch.randn(2, 128, generator=g)
    b = torch.randn(64, 2, generator=g)
    st.save_file({"base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": a,
                  "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": b},
                 str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "base_model_name_or_path": str(base)}))
    (adapter / "tokenizer_config.json").write_text("{}")
    return base, adapter, w, packed, qs, a, b


def test_merge_into_dequantized_base(tmp_path):
    base, adapter, w, packed, qs, a, b = _fake_checkpoint(tmp_path)
    out = tmp_path / "merged"
    export.merge_dequant(adapter, str(base), out)

    from safetensors import safe_open

    index = json.loads((out / "model.safetensors.index.json").read_text())
    shard = out / index["weight_map"]["model.layers.0.mlp.up_proj.weight"]
    with safe_open(str(shard), "pt") as f:
        merged = f.get_tensor("model.layers.0.mlp.up_proj.weight")
        norm = f.get_tensor("model.norm.weight")
        assert not any(k.endswith("absmax") for k in f.keys())  # quant stats are not copied

    expected = (bnb_f.dequantize_4bit(packed, qs).float() + 2.0 * (b @ a)).to(torch.bfloat16)  # scale = 4 / 2
    assert merged.dtype == torch.bfloat16
    assert torch.equal(merged, expected)
    # Merging into the original weights instead gives a different model (the bug this guards against).
    assert not torch.equal(merged, (w.float() + 2.0 * (b @ a)).to(torch.bfloat16))
    assert torch.equal(norm, torch.ones(128, dtype=torch.bfloat16))

    cfg = json.loads((out / "config.json").read_text())
    assert "quantization_config" not in cfg and cfg["torch_dtype"] == "bfloat16"
    assert (out / "tokenizer_config.json").exists()


def test_modelfile_has_tool_template(tmp_path):
    export.write_modelfile(tmp_path, "m.Q8_0.gguf")
    text = (tmp_path / "Modelfile").read_text(encoding="utf-8")
    assert text.startswith("FROM ./m.Q8_0.gguf\nTEMPLATE \"\"\"")
    assert "<tool_call>" in text and "<tool_response>" in text and "PARAMETER num_ctx 8192" in text
