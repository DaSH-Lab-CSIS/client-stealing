"""Residual-peeling loop: subtract accepted sequences and re-search the residual."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from recovery.model import embeddings_are_tied, parallel_source_kind
from recovery.capture import reference_rows
from recovery.search import extract_source_directions, search
from recovery.verify import evaluate, hypothesis_gradient, verify_candidates
from recovery.pipeline import metadata_identity, redact_metrics, write_json


def write_peel_result(path: Path, state: dict[str, Any], snapshot: dict[str, Any], tok) -> dict[str, Any]:
    """Write cumulative residual-peeling recovery JSON."""
    result = {
        "rows": [a["row"] for a in state["accepted"]],
        "coefficients": [a["coefficient"] for a in state["accepted"]],
        "anchor_evidence": state["accepted"],
        "metadata": snapshot["metadata"],
        "history": state["history"],
        "aggregate_observations": 1,
    }
    result["texts"] = tok.batch_decode(result["rows"], skip_special_tokens=True)
    write_json(path, result)
    refs = reference_rows(snapshot)
    if refs is not None:
        result["evaluation"] = evaluate(result["rows"], refs)
        write_json(path, result)
        print(json.dumps(result["evaluation"], indent=2), flush=True)
    else:
        print("Reference rows unavailable; skipped peel evaluation.", flush=True)
    return result


def run_peeling(args, tok, model, snapshot: dict[str, Any], initial: dict[str, Any],
                work_dir: Path, output_path: Path) -> dict[str, Any]:
    """Run/resume residual peeling from the top-level anchor result."""
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.pt"
    if state_path.exists() and not args.force_peel:
        state = torch.load(state_path, weights_only=True)
        if metadata_identity(state["metadata"]) != metadata_identity(snapshot["metadata"]):
            raise ValueError(f"Peeling state {state_path} metadata differs from snapshot")
        print(f"Peeling state exists: {state_path}; completed passes={len(state['history'])}", flush=True)
    else:
        if metadata_identity(initial["metadata"]) != metadata_identity(snapshot["metadata"]):
            raise ValueError("Initial anchor result metadata differs from snapshot")
        state = {"accepted": initial["anchor_evidence"], "history": [], "metadata": snapshot["metadata"]}
        if args.force_peel and state_path.exists():
            print("Ignoring existing peel state because --force-peel was supplied", flush=True)

    original = snapshot["observation"]
    original_norm = original["input"].norm(dim=1)
    if embeddings_are_tied(model) and parallel_source_kind(model) is not None:
        original_bag, _, _ = extract_source_directions(model, original)
    else:
        original_bag = (original_norm > original_norm.max() * 1e-7).nonzero().flatten()
    search_config = {
        "beam": args.beam, "seed_batch": args.seed_batch,
        "lm_weight": args.lm_weight, "source_weight": args.source_weight,
        "mlp_weight": args.mlp_weight, "lookahead": args.lookahead,
        "full_covariance": args.full_covariance,
    }
    verifier_config = {
        "input_tolerance": args.input_tolerance,
        "source_tolerance": args.source_tolerance,
        "min_anchors": args.min_anchors,
        "beta_tolerance": args.beta_tolerance,
        "end_window": args.end_window,
    }

    if not state["accepted"]:
        if not state["history"]:
            state["history"].append({
                "pass": 0,
                "accepted_new": 0,
                "accepted_total": 0,
                "tested_candidates": 0,
                "search_config": search_config,
                "verifier_config": verifier_config,
                "stopped_reason": "no verified sequences in the initial anchor set; residual peeling has nothing to subtract",
            })
            torch.save(state, state_path)
        print("No verified sequences in the initial anchor set; residual peeling has nothing to subtract.", flush=True)
        return write_peel_result(output_path, state, snapshot, tok)

    for iteration in range(len(state["history"]) + 1, args.passes + 1):
        rows = [a["row"] for a in state["accepted"]]
        coefficients = [a["coefficient"] for a in state["accepted"]]
        state_hash = hashlib.sha256(json.dumps([rows, coefficients]).encode()).hexdigest()
        predicted = hypothesis_gradient(model, tok, rows, coefficients, microbatch=args.microbatch)
        residual = {key: original[key] - value for key, value in predicted.items()}
        relative = {key: float(value.norm() / original[key].norm()) for key, value in residual.items()}
        explained = (residual["input"].norm(dim=1) < args.cleanup_tolerance * original_norm) & (original_norm > 0)
        residual["input"] = residual["input"].clone()
        residual["input"][explained] = 0
        residual["source_support"] = original_bag
        metadata = {
            **snapshot["metadata"], "offline_peel_iteration": iteration,
            "peel_state_hash": state_hash, "removed_sequences": len(rows),
            "input_support_cleanup_relative_tolerance": args.cleanup_tolerance,
        }
        residual_snapshot = {"observation": residual, "metadata": metadata}
        if reference_rows(snapshot) is not None:
            residual_snapshot["evaluation"] = snapshot["evaluation"]
        torch.save(residual_snapshot, work_dir / f"residual{iteration}.pt")
        print(
            f"Offline pass {iteration}: removed {len(rows)} sequences; "
            f"cleaned {int(explained.sum())} input rows; residual ratios={relative}",
            flush=True,
        )

        pool_path = work_dir / f"pool{iteration}.pt"
        if pool_path.exists() and not args.force_peel_search:
            pool = torch.load(pool_path, weights_only=True)
            if pool["metadata"] != metadata:
                raise ValueError(f"Saved pool {pool_path} belongs to a different residual")
            if pool.get("offline_search_config") != search_config:
                raise ValueError(f"Saved pool {pool_path} used a different search config")
            print(f"Residual pool exists: {pool_path}", flush=True)
        else:
            checkpoint_path = pool_path.with_suffix(".checkpoint.pt")
            if args.force_peel_search and checkpoint_path.exists():
                checkpoint_path.unlink()
            pool = search(
                model, tok, residual, beam=args.beam, seed_batch=args.seed_batch,
                lm_weight=args.lm_weight, source_weight=args.source_weight,
                mlp_weight=args.mlp_weight, lookahead=args.lookahead,
                full_covariance=args.full_covariance,
                max_length=snapshot["metadata"]["max_length"], keep_atoms=False,
                checkpoint=checkpoint_path,
            )
            pool["metadata"] = metadata
            pool["offline_search_config"] = search_config
            torch.save(pool, pool_path)

        order = torch.tensor(pool["qualities"]).argsort(descending=True).tolist()
        already = set(map(tuple, rows))
        unique = [r for r in dict.fromkeys(tuple(pool["rows"][i]) for i in order) if r not in already][:args.budget]
        new = verify_candidates(
            model, tok, residual, [list(r) for r in unique], batch=args.batch,
            input_tolerance=args.input_tolerance, source_tolerance=args.source_tolerance,
            min_anchors=args.min_anchors, beta_tolerance=args.beta_tolerance,
            end_window=args.end_window, checkpoint=work_dir / f"verification{iteration}.pt",
        )
        state["accepted"].extend(new["anchor_evidence"])
        state["history"].append({
            "pass": iteration,
            "accepted_new": len(new["rows"]),
            "accepted_total": len(state["accepted"]),
            "residual_norm_ratios_before": relative,
            "cleaned_input_rows": int(explained.sum()),
            "tested_candidates": len(unique),
            "search_config": search_config,
            "verifier_config": verifier_config,
        })
        torch.save(state, state_path)
        result = write_peel_result(output_path, state, snapshot, tok)
        if not new["rows"]:
            print("No new sequences passed verification; stopping this configuration.", flush=True)
            return result

    if output_path.exists():
        result = json.loads(output_path.read_text())
        return redact_metrics(result) if args.attack_only else result
    return write_peel_result(output_path, state, snapshot, tok)
