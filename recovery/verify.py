"""Anchor verification against the observed gradient, plus recovery metrics."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from recovery.model import (
    _batch,
    embeddings_are_tied,
    parallel_layers,
    parallel_source_kind,
    selected_parameters,
)
from recovery.search import extract_source_directions


def hypothesis_gradient(model, tok, rows, coefficients, microbatch=8):
    """Sum hypothesized sequence gradients at the supplied positive weights."""
    # _batch() is embedded in this standalone file.
    selected = selected_parameters(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in selected.values():
        p.requires_grad_(True)
    reconstructed = {k: torch.zeros_like(p) for k, p in selected.items()}
    try:
        for start in range(0, len(rows), microbatch):
            part = rows[start:start + microbatch]
            ids, mask = _batch(part, tok.pad_token_id, next(model.parameters()).device)
            logits = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1]
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape(len(part), -1)
            coefficient = loss.new_tensor(coefficients[start:start + len(part)])
            unique, parameter_index = [], {}
            key_index = {}
            for key, parameter in selected.items():
                identity = id(parameter)
                if identity not in parameter_index:
                    parameter_index[identity] = len(unique)
                    unique.append(parameter)
                key_index[key] = parameter_index[identity]
            gradient = torch.autograd.grad((loss.sum(1) * coefficient).sum(), unique)
            for key in selected:
                reconstructed[key].add_(gradient[key_index[key]])
    finally:
        for p in model.parameters():
            p.requires_grad_(False)
    return reconstructed


def verify_gradient(model, tok, observation, rows, coefficients, microbatch=8):
    """Check committed hypotheses on full, unsketched gradient matrices."""
    reconstructed = hypothesis_gradient(model, tok, rows, coefficients, microbatch)
    return {"relative_errors": {key: float((value - observation[key]).norm() / observation[key].norm())
                                for key, value in reconstructed.items()},
            "uses_full_matrices": True, "uses_reference_text": False}


def evaluate(rows, references):
    """One-to-one sequence matching; never concatenate fragments using truth."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    lcs = np.zeros((len(rows), len(references)), dtype=np.int32)
    for i, candidate in enumerate(rows):
        for j, reference in enumerate(references):
            previous = [0] * (len(reference) + 1)
            for token in candidate:
                current = [0]
                for k, other in enumerate(reference):
                    current.append(previous[k] + 1 if token == other else max(current[-1], previous[k + 1]))
                previous = current
            lcs[i, j] = previous[-1]
    ii, jj = linear_sum_assignment(-lcs)
    matched = int(lcs[ii, jj].sum())
    precision = matched / max(1, sum(map(len, rows)))
    recall = matched / max(1, sum(map(len, references)))
    from collections import Counter
    exact = sum((Counter(map(tuple, rows)) & Counter(map(tuple, references))).values())
    false_accepts = max(0, len(rows) - exact)
    return {"exact_sentences": exact, "reference_sentences": len(references),
            "predicted_sentences": len(rows), "exact_sentence_recall": exact / len(references),
            "exact_sentence_precision": exact / max(1, len(rows)),
            "false_accepts": false_accepts,
            "matched_lcs_precision": precision, "matched_lcs_recall": recall,
            "matched_lcs_f1": 2 * precision * recall / max(1e-12, precision + recall)}


@torch.enable_grad()
def candidate_cotangents(model, rows, pad):
    ids, mask = _batch(rows, pad, next(model.parameters()).device)
    inputs = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    captured = []
    handle = None
    supports_source = parallel_source_kind(model) is not None
    need_tied_rows = embeddings_are_tied(model)
    if supports_source:
        def capture_block_output(module, inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        handle = parallel_layers(model)[0].register_forward_hook(capture_block_output)
    try:
        output = model(
            inputs_embeds=inputs, attention_mask=mask, use_cache=False,
            output_hidden_states=need_tied_rows,
        )
        logits = output.logits[:, :-1]
        labels = ids[:, 1:].clone()
        labels[mask[:, 1:] == 0] = -100
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
        if captured:
            gin, source = torch.autograd.grad(loss, (inputs, captured[0]))
            hidden = output.hidden_states[-1][:, :-1].detach() if need_tied_rows else None
            return gin.detach(), source.detach(), hidden, logits.detach() if need_tied_rows else None
        gin, = torch.autograd.grad(loss, (inputs,))
        hidden = output.hidden_states[-1][:, :-1].detach() if need_tied_rows else None
        return gin.detach(), None, hidden, logits.detach() if need_tied_rows else None
    finally:
        if handle is not None:
            handle.remove()


def anchor_good_mask(row, tokens, beta, input_error, source_error, *,
                     observed_input_norm=None, input_floor=0., input_tolerance=3e-3,
                     source_tolerance=3e-3, min_anchors=3, beta_tolerance=2e-3,
                     end_window=5):
    good = (beta > 0) & (input_error < input_tolerance) & (source_error < source_tolerance)
    if observed_input_norm is not None:
        good &= observed_input_norm > input_floor
    if int(good.sum()) < min_anchors:
        return None
    center = beta[good].median()
    good &= (beta - center).abs() < center * beta_tolerance
    if int(good.sum()) < min_anchors:
        return None
    # Require evidence near the end, where a wrong terminal token's gradient
    # contribution cannot be diluted by a long prefix.
    if end_window:
        good_for_tokens = good.to(tokens.device)
        if not (set(tokens[good_for_tokens].tolist()) & set(row[max(0, len(row) - end_window):-1])):
            return None
    return good


def verify_candidates(model, tok, observation, candidates, *, input_tolerance=3e-3,
                      source_tolerance=3e-3, min_anchors=3, beta_tolerance=2e-3,
                      end_window=5, batch=32, checkpoint=None):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bag, source, _ = extract_source_directions(model, observation)
    supports_source = source is not None and parallel_source_kind(model) is not None
    use_tied_embedding_rows = embeddings_are_tied(model)
    source_map = {t: i for i, t in enumerate(bag.tolist())}
    input_floor = observation["input"][bag].norm(dim=1).median() * 1e-4
    fingerprint = hashlib.sha256(json.dumps([
        candidates, input_tolerance, source_tolerance, min_anchors, beta_tolerance, end_window,
        supports_source, use_tied_embedding_rows,
    ]).encode())
    fingerprint.update(observation["input"].detach().float().cpu().numpy().tobytes())
    if "mlp0" in observation:
        fingerprint.update(observation["mlp0"].detach().float().cpu().numpy().tobytes())
    fingerprint = fingerprint.hexdigest()
    accepted, best, fits, first = [], [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        saved = torch.load(checkpoint, weights_only=True)
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Verifier checkpoint differs; choose a new output/checkpoint path")
        accepted, fits, first = saved["accepted"], saved["fits"], saved["processed"]
        best = [sorted(f["input_error"])[min_anchors - 1] for f in fits]
    started = time.monotonic()
    for start in range(first, len(candidates), batch):
        part = candidates[start:start + batch]
        inputs, cotangents, hidden_states, logits = candidate_cotangents(model, part, tok.pad_token_id)
        for ri, row in enumerate(part):
            tokens = sorted(set(row[:-1]) & source_map.keys())
            if len(tokens) < min_anchors:
                continue
            token_tensor = torch.tensor(tokens, device=inputs.device)
            row_tensor = torch.tensor(row, device=inputs.device)
            predicted_input = torch.stack([
                inputs[ri, :len(row)][row_tensor == t].sum(0) for t in tokens
            ]).double()
            if use_tied_embedding_rows:
                # For tied embeddings (GPT-2), the observed parameter row is
                # input-embedding cotangent plus LM-head row gradient.
                length = len(row)
                hidden = hidden_states[ri, :length - 1].double()
                probabilities = logits[ri, :length - 1].softmax(-1)[:, token_tensor].double()
                predicted_output = probabilities.T @ hidden
                labels = torch.tensor(row[1:], device=hidden.device)
                for ti, token in enumerate(tokens):
                    predicted_output[ti] -= hidden[labels == token].sum(0)
                predicted_anchor = predicted_input + predicted_output
            else:
                predicted_anchor = predicted_input
            observed_input = observation["input"][token_tensor].double()
            norm2 = predicted_anchor.square().sum(1).clamp_min(1e-30)
            beta = (predicted_anchor * observed_input).sum(1) / norm2
            input_error = (observed_input - beta[:, None] * predicted_anchor).norm(dim=1) / observed_input.norm(dim=1).clamp_min(1e-30)
            if supports_source:
                predicted_source = torch.stack([
                    cotangents[ri, :len(row)][row_tensor == t].sum(0) for t in tokens
                ]).double()
                observed_source = source[torch.tensor(
                    [source_map[t] for t in tokens], device=source.device
                )].double()
                source_error = (observed_source - beta[:, None] * predicted_source).norm(dim=1) / observed_source.norm(dim=1).clamp_min(1e-30)
            else:
                source_error = torch.zeros_like(input_error)
            good = anchor_good_mask(row, token_tensor, beta, input_error, source_error,
                                    observed_input_norm=observed_input.norm(dim=1),
                                    input_floor=input_floor, input_tolerance=input_tolerance,
                                    source_tolerance=source_tolerance, min_anchors=min_anchors,
                                    beta_tolerance=beta_tolerance, end_window=end_window)
            best.append(float(input_error.sort().values[min_anchors - 1]))
            fits.append({"candidate": start + ri, "tokens": tokens, "beta": beta.tolist(),
                         "input_error": input_error.tolist(), "source_error": source_error.tolist()})
            if good is not None:
                coefficient = ((predicted_anchor[good] * observed_input[good]).sum() /
                               predicted_anchor[good].square().sum())
                accepted.append({"row": row, "coefficient": float(coefficient),
                                 "anchor_tokens": token_tensor[good].tolist(),
                                 "input_relative_errors": input_error[good].tolist(),
                                 "source_relative_errors": source_error[good].tolist(),
                                 "used_source_anchors": supports_source,
                                 "used_tied_embedding_rows": use_tied_embedding_rows})
        if start % (8 * batch) == 0 or start + batch >= len(candidates):
            print(f"Verified {min(start + batch, len(candidates))}/{len(candidates)}; "
                  f"accepted={len(accepted)}; seconds={time.monotonic() - started:.1f}", flush=True)
            if checkpoint is not None:
                temporary = Path(checkpoint).with_suffix(".tmp")
                torch.save({"fits": fits, "accepted": accepted, "fingerprint": fingerprint,
                            "processed": start + len(part)}, temporary)
                temporary.replace(checkpoint)
    return {"rows": [a["row"] for a in accepted], "coefficients": [a["coefficient"] for a in accepted],
            "anchor_evidence": accepted, "tested_candidates": len(candidates),
            "input_tolerance": input_tolerance, "source_tolerance": source_tolerance,
            "minimum_anchors": min_anchors, "beta_tolerance": beta_tolerance,
            "end_window": end_window, "used_source_anchors": supports_source,
            "used_tied_embedding_rows": use_tied_embedding_rows,
            "second_best_input_residual_min": min(best, default=None)}


def topk_counts(pool, references, budgets, exclude=()):
    order = torch.tensor(pool["qualities"]).argsort(descending=True).tolist()
    seen, ranked = set(), []
    for index in order:
        row = tuple(pool["rows"][index])
        if row not in seen:
            seen.add(row)
            ranked.append(row)
    ref_counter = Counter(map(tuple, references))
    ref_counter.subtract(Counter(map(tuple, exclude)))
    ref_counter = +ref_counter
    counts = {}
    for budget in budgets:
        counts[str(budget)] = sum((Counter(ranked[:budget]) & ref_counter).values())
    return ranked, counts


def base_metadata(metadata):
    # PEEL_METADATA_KEYS is imported lazily here to avoid an import cycle:
    # pipeline imports verify (topk_counts, verify_candidates, evaluate), so
    # verify must not import pipeline at module load time.
    from recovery.pipeline import PEEL_METADATA_KEYS
    return {key: value for key, value in metadata.items() if key not in PEEL_METADATA_KEYS}
