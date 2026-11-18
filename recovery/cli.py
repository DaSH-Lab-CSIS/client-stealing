"""End-to-end one-round text recovery from an aggregated FL gradient.

This package contains the gradient-capture, aggregate subtraction, candidate
search, anchor verification, residual-peeling, and reporting stages.

External runtime dependencies are still the scientific/model packages used by
the experiment: torch, transformers, numpy, and scipy.  The model is loaded from
the local Hugging Face cache with local_files_only=True.

The source-fingerprint method has been validated on untied GPT-NeoX/Pythia
causal LMs with parallel residual blocks. Use --model-name to switch locally
cached checkpoints. Other causal LMs use an experimental embedding-only
fallback; executing that path does not establish high recovery or equivalent
verifier behavior. See docs/ for measured limitations.

Pipeline stages:

1. Capture a true weighted client aggregate, subtract only the acting client's
   known contribution, and save the unknown-client aggregate snapshot.
2. Generate candidate sentences from token/source fingerprints in the aggregate.
3. Write oracle pool diagnostics after committing the candidate pool.
4. Verify candidates by recomputing exact gradient cotangents and checking
   whether token anchors match the observed aggregate at a common positive scale.
5. Repeatedly peel accepted candidates from the aggregate residual and rerun
   search + verification from durable checkpoints.

The script is resumable. Existing snapshots, pools, verifier checkpoints, and
peeling state are reused unless a matching --force-* flag is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from recovery.model import load_public_model
from recovery.pipeline import (
    load_or_capture_snapshot,
    run_or_load_anchor,
    run_or_load_search,
    summarize,
    tolerance_tag,
    write_pool_diagnostic,
)
from recovery.peel import run_peeling


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-data", type=Path,
                    help="Client JSON used only to simulate/capture an aggregate gradient.")
    ap.add_argument("--snapshot", type=Path,
                    help="Precomputed aggregate snapshot. Use this for attack-only recovery from one observed gradient.")
    ap.add_argument("--attack-only", action="store_true",
                    help="Refuse client data, redact references, and skip oracle/evaluation metrics.")
    ap.add_argument("--model-name", default="EleutherAI/pythia-160m",
                    help="Locally cached GPT-NeoX/Pythia checkpoint to use for capture and recovery.")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Load a cached checkpoint's custom modeling code (needed by StableLM Alpha v2).")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory where all durable artifacts for this run are stored.")
    ap.add_argument(
        "--passes", type=int, default=24,
        help="Maximum peel passes; recovery stops early on the first zero-yield pass.",
    )
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--seed-batch", type=int, default=16)
    ap.add_argument("--seed-limit", type=int, default=0,
                    help="0 means search every live start token; positive values are smoke-test only.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=24)
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--lm-weight", type=float, default=0.02)
    ap.add_argument("--source-weight", type=float, default=0.0) # disabled, was for experimentation
    ap.add_argument("--mlp-weight", type=float, default=0.0) # disabled, was for experimentation
    ap.add_argument("--lookahead", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="float32 is the reference precision; reduced precision is an explicit ablation.")
    ap.add_argument("--full-covariance", action="store_true", default=True,
                    help="Use full-covariance whitening. Default on for current experiments.")
    ap.add_argument("--no-full-covariance", dest="full_covariance", action="store_false")
    ap.add_argument("--input-tolerance", type=float, default=0.005)
    ap.add_argument("--source-tolerance", type=float, default=0.005)
    ap.add_argument("--min-anchors", type=int, default=3)
    ap.add_argument("--beta-tolerance", type=float, default=2e-3)
    ap.add_argument("--end-window", type=int, default=5)
    ap.add_argument("--cleanup-tolerance", type=float, default=0.003)
    ap.add_argument("--skip-peel", action="store_true")
    ap.add_argument("--force-capture", action="store_true")
    ap.add_argument("--force-search", action="store_true")
    ap.add_argument("--force-anchor", action="store_true")
    ap.add_argument("--force-peel", action="store_true")
    ap.add_argument("--force-peel-search", action="store_true")
    args = ap.parse_args()
    if args.attack_only and args.client_data is not None:
        ap.error("--attack-only refuses --client-data; pass --snapshot or use an existing high_recovery128.pt in --run-dir.")
    if args.snapshot is None and args.client_data is None and not (args.run_dir / "high_recovery128.pt").exists():
        ap.error("provide --client-data to capture a new aggregate, or --snapshot/--run-dir with an existing aggregate snapshot")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Custom public checkpoints (StableLM Alpha v2) materialize their cached
    # Python module here.  Keep it inside the durable run directory rather than
    # relying on a potentially read-only home cache.
    os.environ.setdefault("HF_MODULES_CACHE", str((args.run_dir / "hf_modules").resolve()))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    args.run_dir.mkdir(parents=True, exist_ok=True)

    pool_tag = f"seed{args.seed_limit}" if args.seed_limit else "allstarts"
    tol_tag = tolerance_tag(args.input_tolerance)
    budget_tag = f"b{args.budget // 1000}k" if args.budget % 1000 == 0 else f"b{args.budget}"
    snapshot_path = args.run_dir / "high_recovery128.pt"
    pool_path = args.run_dir / f"pool_fullcov_b{args.beam}_{pool_tag}.pt"
    report_prefix = "attack_only_" if args.attack_only else ""
    pool_report_path = args.run_dir / f"{report_prefix}pool_fullcov_b{args.beam}_{pool_tag}_report.json"
    anchor_path = args.run_dir / f"anchor_fullcov_b{args.beam}_{pool_tag}_budget{args.budget}_tol{tol_tag}.json"
    peel_work = args.run_dir / f"offline_peel_from_anchor_fullcov_b{args.beam}_{budget_tag}"
    peel_output = args.run_dir / f"offline_peel_from_anchor_fullcov_b{args.beam}_{budget_tag}.json"
    summary_path = args.run_dir / (
        "attack_only_standalone_end_to_end_summary.json"
        if args.attack_only else "standalone_end_to_end_summary.json"
    )

    print("Artifact plan:", flush=True)
    for label, path in [
        ("snapshot", snapshot_path), ("pool", pool_path), ("pool report", pool_report_path),
        ("anchor", anchor_path), ("peel work", peel_work), ("peel output", peel_output),
        ("summary", summary_path),
    ]:
        print(f"  {label}: {path}", flush=True)
    if args.attack_only:
        print("Attack-only mode: no client JSON is loaded and reference/evaluation rows are redacted.", flush=True)

    tok, model = load_public_model(
        args.threads, args.model_name, args.device, args.model_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    snapshot = load_or_capture_snapshot(args, tok, model, snapshot_path)
    pool = run_or_load_search(args, tok, model, snapshot, pool_path)
    pool_report = write_pool_diagnostic(snapshot, pool, pool_path, pool_report_path,
                                        [1000, 5000, 10000, 20000, 50000, 100000])
    anchor = run_or_load_anchor(args, tok, model, snapshot, pool, pool_path, anchor_path)
    peel = None
    if not args.skip_peel:
        peel = run_peeling(args, tok, model, snapshot, anchor, peel_work, peel_output)

    summary = summarize(summary_path, snapshot, pool_report, anchor, peel)
    print("Final summary:", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}, indent=2), flush=True)
