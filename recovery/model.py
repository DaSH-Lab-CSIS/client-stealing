"""Model architecture abstraction: parallel-residual adapters and public-checkpoint loading."""
from __future__ import annotations

from typing import Any

import torch


def tensors_to_device(obj, device):
    """Move nested tensor payloads while leaving JSON-like metadata intact."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: tensors_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [tensors_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(tensors_to_device(value, device) for value in obj)
    return obj


def configure_cuda_numerics() -> None:
    """Use deterministic/eager CUDA math consistent across capture and verify."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _batch(rows, pad, device):
    ids = torch.full((len(rows), max(map(len, rows))), pad,
                     dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        mask[i, :len(row)] = 1
    return ids, mask


def is_neox_parallel_model(model) -> bool:
    """Whether the strong Pythia/GPT-NeoX source-fingerprint path is available."""
    if not hasattr(model, "gpt_neox"):
        return False
    layers = getattr(model.gpt_neox, "layers", [])
    if len(layers) < 2:
        return False
    required = [
        getattr(layers[0].mlp, "dense_h_to_4h", None),
        getattr(layers[0].mlp, "dense_4h_to_h", None),
        getattr(layers[1].mlp, "dense_h_to_4h", None),
        getattr(layers[1].mlp, "dense_4h_to_h", None),
        getattr(layers[1].attention, "query_key_value", None),
    ]
    return all(value is not None for value in required) and bool(getattr(model.config, "use_parallel_residual", False))


def parallel_source_kind(model) -> str | None:
    """Return the exact token-only first-block MLP dictionary implemented by ``model``.

    The recovery source is architecture-independent once a block exposes a
    parallel residual update

        x_1 = x_0 + Attention(Norm(x_0)) + W_out A(Norm(x_0)).

    The gradient of ``W_out`` factorizes as ``G = Delta.T @ A`` and ``A`` is a
    public function of the input token alone.  The adapter below recognizes the
    three concrete layouts used in the experiments; it deliberately refuses a
    sequential block rather than silently using a context-contaminated
    dictionary.
    """
    if is_neox_parallel_model(model):
        return "gpt_neox"
    if getattr(model.config, "model_type", None) == "falcon":
        transformer = getattr(model, "transformer", None)
        layers = getattr(transformer, "h", [])
        if (bool(getattr(model.config, "parallel_attn", False)) and len(layers) >= 2
                and hasattr(layers[0], "input_layernorm")
                and hasattr(layers[0].mlp, "dense_h_to_4h")
                and hasattr(layers[0].mlp, "dense_4h_to_h")
                and hasattr(layers[1].self_attention, "query_key_value")):
            return "falcon"
    if getattr(model.config, "model_type", None) == "stablelm_alpha":
        transformer = getattr(model, "transformer", None)
        layers = getattr(transformer, "layers", [])
        if (len(layers) >= 2 and hasattr(layers[0], "norm")
                and hasattr(layers[0].mlp, "gate_proj")
                and hasattr(layers[0].mlp, "out_proj")
                and hasattr(layers[1].attention, "qkv_proj")):
            return "stablelm_alpha"
    return None


def parallel_layers(model):
    kind = parallel_source_kind(model)
    if kind == "gpt_neox":
        return model.gpt_neox.layers
    if kind == "falcon":
        return model.transformer.h
    if kind == "stablelm_alpha":
        return model.transformer.layers
    raise ValueError("model has no certified parallel-residual source adapter")


def parallel_base_model(model):
    kind = parallel_source_kind(model)
    if kind == "gpt_neox":
        return model.gpt_neox
    if kind in {"falcon", "stablelm_alpha"}:
        return model.transformer
    raise ValueError("model has no certified parallel-residual source adapter")


def parallel_mlp_output_weight(model, layer: int):
    block = parallel_layers(model)[layer]
    kind = parallel_source_kind(model)
    if kind in {"gpt_neox", "falcon"}:
        return block.mlp.dense_4h_to_h.weight
    if kind == "stablelm_alpha":
        return block.mlp.out_proj.weight
    raise AssertionError(kind)


def parallel_attention_weight(model, layer: int):
    block = parallel_layers(model)[layer]
    kind = parallel_source_kind(model)
    if kind == "gpt_neox":
        return block.attention.query_key_value.weight
    if kind == "falcon":
        return block.self_attention.query_key_value.weight
    if kind == "stablelm_alpha":
        return block.attention.qkv_proj.weight
    raise AssertionError(kind)


def parallel_token_activation(model, embedding: torch.Tensor) -> torch.Tensor:
    """Public token-only activation multiplying the first MLP output matrix."""
    block = parallel_layers(model)[0]
    kind = parallel_source_kind(model)
    if kind == "gpt_neox":
        hidden = block.post_attention_layernorm(embedding)
        return block.mlp.act(block.mlp.dense_h_to_4h(hidden))
    if kind == "falcon":
        # Falcon-7B's parallel_attn block feeds the same normalized residual to
        # attention and MLP.  new_decoder_architecture variants may expose an
        # explicit ln_mlp; use it when present.
        normalizer = getattr(block, "ln_mlp", None) or block.input_layernorm
        hidden = normalizer(embedding)
        return block.mlp.act(block.mlp.dense_h_to_4h(hidden))
    if kind == "stablelm_alpha":
        hidden = block.norm(embedding)
        ff, gate = block.mlp.gate_proj(hidden).chunk(2, dim=-1)
        return ff * block.mlp.act(gate)
    raise AssertionError(kind)


def embeddings_are_tied(model) -> bool:
    return model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def selected_parameters(model) -> dict[str, torch.Tensor]:
    """Gradient matrices the recovery pipeline observes for this architecture."""
    selected = {
        "input": model.get_input_embeddings().weight,
        "output": model.get_output_embeddings().weight,
    }
    if parallel_source_kind(model) is not None:
        selected.update({
            "attention": parallel_attention_weight(model, 1),
            "mlp0": parallel_mlp_output_weight(model, 0),
            "mlp1": parallel_mlp_output_weight(model, 1),
        })
    return selected


def model_support(model) -> dict[str, Any]:
    """Human-readable capability summary stored in run metadata."""
    return {
        "model_type": getattr(model.config, "model_type", model.__class__.__name__),
        "neox_parallel_source": is_neox_parallel_model(model),
        "parallel_source_kind": parallel_source_kind(model),
        "parallel_source": parallel_source_kind(model) is not None,
        "generic_input_output": True,
        "tied_input_output_embeddings": embeddings_are_tied(model),
        "selected_gradient_keys": list(selected_parameters(model).keys()),
    }


def require_supported_model(model) -> None:
    """Fail only when the model lacks the minimum generic causal-LM interface."""
    missing = []
    if model.get_input_embeddings() is None:
        missing.append("input embeddings")
    if model.get_output_embeddings() is None:
        missing.append("output embeddings / LM head")
    if missing:
        raise ValueError(
            "This standalone recovery implementation needs a causal LM with exposed "
            "input embeddings and output embeddings. Missing: " + ", ".join(missing)
        )
    if parallel_source_kind(model) is None:
        print(
            "Using generic model fallback: certified parallel source/MLP signals are disabled, "
            "so recovery may be substantially weaker.",
            flush=True,
        )


def load_public_model(threads: int, model_name: str, device_name: str, model_dtype: str,
                      trust_remote_code: bool = False):
    """Load the public model/tokenizer once for all stages."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    configure_cuda_numerics()
    tok = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, trust_remote_code=trust_remote_code,
    )
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).eval().to(device)
    require_supported_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"Loaded {model_name} on {device} as {dtype}; "
        f"parallel_source={parallel_source_kind(model)} hidden={model.config.hidden_size} "
        f"tied={embeddings_are_tied(model)}",
        flush=True,
    )
    return tok, model
