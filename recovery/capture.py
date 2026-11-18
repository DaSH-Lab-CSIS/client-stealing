"""FL gradient capture, known-client aggregate subtraction, and client payload loading."""
from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn.functional as F

from recovery.model import _batch, model_support, selected_parameters


def subtract_known_client(aggregate, own, own_weight):
    """Return the normalized residual of a known linear, weighted aggregation.

    Applies to raw gradients or additive updates in the SAME units. Applying it
    to an Adam/FedAvg delta does not turn that delta into a raw gradient.
    """
    if not 0 <= own_weight < 1:
        raise ValueError("own_weight must be in [0, 1)")
    if not aggregate or aggregate.keys() != own.keys():
        raise ValueError("Aggregate and own contribution must have matching keys")
    residual = {}
    for name, value in aggregate.items():
        other = own[name]
        if value.shape != other.shape or value.device != other.device:
            raise ValueError(f"Shape/device mismatch for {name}")
        if not value.is_floating_point() or not other.is_floating_point():
            raise ValueError("Contributions must be floating point tensors")
        if not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f"Nonfinite contribution: {name}")
        residual[name] = ((value.double() - own_weight * other.double()) /
                          (1 - own_weight)).to(value.dtype)
    return residual


def capture(model, tok, clients, max_length=24, microbatch=8, model_name=None):
    selected = selected_parameters(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in selected.values():
        p.requires_grad_(True)
    weights = [c["weight"] / sum(v["weight"] for v in clients) for c in clients]
    aggregate, own, refs = None, None, []
    for ci, (client, weight) in enumerate(zip(clients, weights)):
        rows = [tok(t, truncation=True, max_length=max_length).input_ids for t in client["texts"]]
        count = sum(len(row) - 1 for row in rows)
        gradient = {k: torch.zeros_like(p) for k, p in selected.items()}
        for start in range(0, len(rows), microbatch):
            part = rows[start:start + microbatch]
            ids, mask = _batch(part, tok.pad_token_id, next(model.parameters()).device)
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            logits = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                   labels.reshape(-1), reduction="sum") / count
            # Tied input/output embeddings are one parameter.  Differentiate it
            # once and copy the combined row gradient into both named views.
            unique, parameter_index = [], {}
            key_index = {}
            for key, parameter in selected.items():
                identity = id(parameter)
                if identity not in parameter_index:
                    parameter_index[identity] = len(unique)
                    unique.append(parameter)
                key_index[key] = parameter_index[identity]
            grads = torch.autograd.grad(loss, unique)
            for key in selected:
                gradient[key].add_(grads[key_index[key]])
        if ci == 0:
            own = gradient
        else:
            refs.extend(rows)
        if aggregate is None:
            aggregate = {k: weight * g for k, g in gradient.items()}
        else:
            for k, g in gradient.items():
                aggregate[k].add_(g, alpha=weight)
        print(f"Captured client {ci + 1}/{len(clients)}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    observation = subtract_known_client(aggregate, own, weights[0])
    return {"observation": observation, "evaluation": {"rows": refs},
            "metadata": {"model": model_name or getattr(model.config, "_name_or_path", model.__class__.__name__),
                         "max_length": max_length,
                         "aggregate_observations": 1, "microbatch": microbatch,
                         "own_weight": weights[0], "local_optimizer_steps": 1,
                         "model_modified": False,
                         "model_support": model_support(model)}}


def load_client_payload(path):
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return payload["clients"], payload.get("metadata", {})
    return payload, {}


def reference_rows(snapshot: dict[str, Any]) -> list[Any] | None:
    """Return held-out references only when metric reporting is explicitly available."""
    rows = snapshot.get("evaluation", {}).get("rows")
    return rows if rows else None


def redact_references(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove reference rows so recovery code cannot accidentally read them."""
    redacted = dict(snapshot)
    redacted.pop("evaluation", None)
    return redacted
