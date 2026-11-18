"""Run orchestration: snapshot capture/load, search/anchor runners, IO helpers, and summary."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from recovery.model import tensors_to_device
from recovery.capture import capture, load_client_payload, redact_references, reference_rows
from recovery.search import ranked_unique_candidates, search
from recovery.verify import evaluate, topk_counts, verify_candidates


PEEL_METADATA_KEYS = {
    "offline_peel_iteration", "peel_state_hash", "removed_sequences",
    "input_support_cleanup_relative_tolerance",
}


def metadata_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata before comparing stages.

    Peeling residual snapshots add pass-specific metadata.  Early snapshots may
    also omit empty client metadata.  These fields should not make a saved pool
    or verifier result look incompatible with its parent observation.
    """
    return {
        key: value for key, value in metadata.items()
        if key not in PEEL_METADATA_KEYS and not (key == "client_data_metadata" and value == {})
    }


def tolerance_tag(value: float) -> str:
    """Format 0.005 as '005' for compatibility with existing artifact names."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("0.", "").replace(".", "p") or "0"


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def redact_metrics(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop metric-only fields when running in attack-only mode."""
    if result is None:
        return None
    redacted = dict(result)
    redacted.pop("evaluation", None)
    return redacted


def load_or_capture_snapshot(args, tok, model, snapshot_path: Path) -> dict[str, Any]:
    """Compute or load the acting-client-subtracted aggregate snapshot."""
    model_revision = getattr(model.config, "_commit_hash", None)

    if args.snapshot is not None:
        snapshot = torch.load(args.snapshot, weights_only=True, map_location=next(model.parameters()).device)
        snapshot["observation"] = tensors_to_device(snapshot["observation"], next(model.parameters()).device)
        if snapshot["metadata"].get("model") != args.model_name:
            raise ValueError(
                f"Snapshot {args.snapshot} was built for {snapshot['metadata'].get('model')}; "
                f"rerun with --model-name {snapshot['metadata'].get('model')}"
            )
        if snapshot["metadata"].get("model_revision") != model_revision:
            raise ValueError(f"Snapshot {args.snapshot} was built from a different model revision")
        if args.attack_only:
            snapshot = redact_references(snapshot)
        if args.snapshot.resolve() != snapshot_path.resolve() and (args.force_capture or not snapshot_path.exists()):
            torch.save(snapshot, snapshot_path)
            print(f"Copied supplied aggregate snapshot to {snapshot_path}", flush=True)
        else:
            print(f"Loaded supplied aggregate snapshot: {args.snapshot}", flush=True)
        return snapshot

    if snapshot_path.exists() and not args.force_capture:
        snapshot = torch.load(snapshot_path, weights_only=True, map_location=next(model.parameters()).device)
        snapshot["observation"] = tensors_to_device(snapshot["observation"], next(model.parameters()).device)
        if snapshot["metadata"].get("model") != args.model_name:
            raise ValueError(
                f"Snapshot {snapshot_path} was built for {snapshot['metadata'].get('model')}; "
                f"use a separate --run-dir or rerun with --model-name {snapshot['metadata'].get('model')}"
            )
        if args.client_data is not None:
            client_digest = hashlib.sha256(args.client_data.read_bytes()).hexdigest()
            if snapshot["metadata"].get("client_data_sha256") != client_digest:
                raise ValueError(f"Snapshot {snapshot_path} was built from different client data")
        if snapshot["metadata"].get("model_revision") != model_revision:
            raise ValueError(f"Snapshot {snapshot_path} was built from a different model revision")
        if args.attack_only:
            snapshot = redact_references(snapshot)
        print(f"Snapshot exists: {snapshot_path}", flush=True)
        return snapshot

    if args.client_data is None:
        raise ValueError("Provide --client-data to simulate/capture the FL round, or --snapshot for attack-only recovery.")

    client_digest = hashlib.sha256(args.client_data.read_bytes()).hexdigest()
    clients, client_metadata = load_client_payload(args.client_data)
    started = time.monotonic()
    snapshot = capture(
        model, tok, clients, max_length=args.max_length,
        microbatch=args.microbatch, model_name=args.model_name,
    )
    snapshot["metadata"]["client_data_sha256"] = client_digest
    if client_metadata:
        snapshot["metadata"]["client_data_metadata"] = client_metadata
    snapshot["metadata"]["model_revision"] = model_revision
    torch.save(snapshot, snapshot_path)
    print(f"Saved immutable observation to {snapshot_path} in {time.monotonic() - started:.1f}s", flush=True)
    return snapshot


def run_or_load_search(args, tok, model, snapshot: dict[str, Any], pool_path: Path) -> dict[str, Any]:
    """Generate or load the candidate sentence pool."""
    if pool_path.exists() and not args.force_search:
        pool = torch.load(pool_path, weights_only=True)
        if metadata_identity(pool["metadata"]) != metadata_identity(snapshot["metadata"]):
            raise ValueError(f"Pool {pool_path} metadata differs from snapshot")
        print(f"Pool exists: {pool_path}", flush=True)
        return pool

    checkpoint_path = pool_path.with_suffix(".checkpoint.pt")
    if args.force_search and checkpoint_path.exists():
        checkpoint_path.unlink()
    started = time.monotonic()
    pool = search(
        model, tok, snapshot["observation"], beam=args.beam,
        seed_batch=args.seed_batch, lm_weight=args.lm_weight,
        source_weight=args.source_weight, mlp_weight=args.mlp_weight,
        lookahead=args.lookahead, full_covariance=args.full_covariance,
        seed_limit=args.seed_limit, keep_atoms=False,
        max_length=snapshot["metadata"]["max_length"], checkpoint=checkpoint_path,
    )
    pool["metadata"] = snapshot["metadata"]
    torch.save(pool, pool_path)
    print(f"Saved candidate pool to {pool_path} in {time.monotonic() - started:.1f}s", flush=True)
    return pool


def write_pool_diagnostic(snapshot: dict[str, Any], pool: dict[str, Any], pool_path: Path,
                          report_path: Path, budgets: list[int]) -> dict[str, Any]:
    """Write oracle pool-recall diagnostics after the pool is committed."""
    refs = reference_rows(snapshot)
    order = torch.tensor(pool["qualities"]).argsort(descending=True).tolist()
    ranked = list(dict.fromkeys(tuple(pool["rows"][index]) for index in order))
    result = {
        "pool": str(pool_path),
        "snapshot": str(report_path.parent / "high_recovery128.pt"),
        "metadata": snapshot["metadata"],
        "search": pool.get("info", {}),
        "excluded_recovery_files": [],
        "excluded_rows": 0,
        "candidate_count": len(pool["rows"]),
        "unique_candidate_count": len(ranked),
        "reference_metrics_available": refs is not None,
    }
    if refs is not None:
        _, counts = topk_counts(pool, refs, budgets)
        candidate_set = set(map(tuple, pool["rows"]))
        ref_counter = Counter(map(tuple, refs))
        result.update({
            "remaining_reference_sentences": sum(ref_counter.values()),
            "oracle_pool_exact_sentences": sum(count for row, count in ref_counter.items() if row in candidate_set),
            "topk_unique_exact_sentences": counts,
        })
    write_json(report_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2), flush=True)
    return result


def run_or_load_anchor(args, tok, model, snapshot: dict[str, Any], pool: dict[str, Any],
                       pool_path: Path, anchor_path: Path) -> dict[str, Any]:
    """Run or load the top-level anchor verifier result."""
    if anchor_path.exists() and not args.force_anchor:
        result = json.loads(anchor_path.read_text())
        if metadata_identity(result["metadata"]) != metadata_identity(snapshot["metadata"]):
            raise ValueError(f"Anchor result {anchor_path} metadata differs from snapshot")
        if args.attack_only:
            result = redact_metrics(result)
        print(f"Anchor result exists: {anchor_path}", flush=True)
        return result

    unique = ranked_unique_candidates([pool], args.budget)
    started = time.monotonic()
    checkpoint = anchor_path.with_suffix(".verification.pt")
    if args.force_anchor and checkpoint.exists():
        checkpoint.unlink()
    result = verify_candidates(
        model, tok, snapshot["observation"], unique, batch=args.batch,
        input_tolerance=args.input_tolerance, source_tolerance=args.source_tolerance,
        min_anchors=args.min_anchors, beta_tolerance=args.beta_tolerance,
        end_window=args.end_window, checkpoint=checkpoint,
    )
    result["metadata"] = snapshot["metadata"]
    result["pools"] = [str(pool_path)]
    result["texts"] = tok.batch_decode(result["rows"], skip_special_tokens=True)
    result["seconds"] = time.monotonic() - started
    write_json(anchor_path, result)

    # Reference rows are opened only after the committed recovery output exists.
    refs = reference_rows(snapshot)
    if refs is not None:
        result["evaluation"] = evaluate(result["rows"], refs)
        write_json(anchor_path, result)
        print(json.dumps(result["evaluation"], indent=2), flush=True)
    else:
        print("Reference rows unavailable; skipped anchor evaluation.", flush=True)
    return result


def summarize(summary_path: Path, snapshot: dict[str, Any], pool_report: dict[str, Any],
              anchor: dict[str, Any], peel: dict[str, Any] | None) -> dict[str, Any]:
    """Write a compact run summary for dashboards and comparisons."""
    summary = {
        "metadata": snapshot["metadata"],
        "pool_report": {k: v for k, v in pool_report.items() if k != "metadata"},
        "anchor_evaluation": anchor.get("evaluation"),
        "anchor_rows": len(anchor.get("rows", [])),
    }
    if peel is not None:
        summary["peel_evaluation"] = peel.get("evaluation")
        summary["peel_history"] = peel.get("history", [])
        summary["peel_rows"] = len(peel.get("rows", []))
    write_json(summary_path, summary)
    return summary
