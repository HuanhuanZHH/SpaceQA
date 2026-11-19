"""Command-line entry points; rule computation stays in :mod:`engine`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> None:
    """Run offline checks/audits, or explicitly request an API explanation."""
    from . import engine

    parser = argparse.ArgumentParser(
        prog="spaceqa", description="SpaceQA Agent: evidence-first dataset quality evaluation"
    )
    parser.add_argument("--version", action="version", version=f"SpaceQA Agent {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="evaluate one package or a publish root using rules")
    scan.add_argument("--input", required=True, type=Path)
    scan.add_argument("--output", required=True, type=Path)
    scan.add_argument("--contract", type=Path, help="versioned DEL-4 target protocol JSON")
    scan.add_argument("--seed", type=int, default=20261021)
    scan.add_argument("--checks", default=",".join(engine.ALL_CHECKS), help="comma-separated check IDs")
    scan.add_argument("--backend-profile", choices=("enhanced", "legacy"), default="enhanced",
                      help="merge family-aware backend signals into formal findings; legacy preserves the pre-migration rule path")
    audit_parser = commands.add_parser("audit", help="independently recompute scores and verify evidence")
    audit_parser.add_argument("run_dir", type=Path)
    explanation = commands.add_parser("explain", help="explain frozen results through an optional API")
    explanation.add_argument("--case", type=Path, help="current case and model results as JSON")
    explanation.add_argument("--run-dir", type=Path, help="read existing rule findings")
    explanation.add_argument("--finding-id", action="append", default=[])
    explanation.add_argument("--limit", type=int, default=20)
    explanation.add_argument("--image", type=Path, action="append", default=[])
    explanation.add_argument("--base-url")
    explanation.add_argument("--model")
    explanation.add_argument("--timeout", type=float, default=60)
    explanation.add_argument("--retry-limit", type=int, default=1)
    explanation.add_argument("--output", type=Path, required=True)
    prior = commands.add_parser("check-priors", help="run current-input experimental prior checks")
    prior.add_argument("--input", type=Path, required=True)
    prior.add_argument("--family", required=True, choices=("AIA_AR_sample", "HMI_AR_sample",
                       "CME_C2_sample", "EPB_seg_sample", "DMSP_SSUSI", "ULF_sample"))
    prior.add_argument("--output", type=Path, required=True, help="new JSONL result outside the input")
    prior.add_argument("--trust-ulf-dateobs", action="store_true")
    prior.add_argument("--ulf-duration-convention", choices=("inclusive", "elapsed", "either"), default="either")
    prior.add_argument("--backend-profile", choices=("enhanced", "legacy"), default="enhanced",
                       help="family-aware AIA94-derived observations; legacy keeps the previous dialog profile")
    web = commands.add_parser("serve", help="start the local web UI and background evaluation API")
    web.add_argument("--host", default="127.0.0.1", help="loopback address; access remotely through SSH forwarding")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--data-root", action="append", required=True, type=Path,
                     help="allowed dataset root; repeat for multiple roots")
    web.add_argument("--run-root", action="append", default=[], type=Path,
                     help="existing audited run or parent directory; repeat as needed")
    web.add_argument("--state-dir", required=True, type=Path,
                     help="persistent job state and new outputs outside all input roots")
    args = parser.parse_args(argv)

    if args.command == "serve":
        from .web_service import serve

        try:
            serve(host=args.host, port=args.port, data_roots=args.data_root,
                  run_roots=args.run_root, state_dir=args.state_dir)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return

    if args.command == "check-priors":
        from . import priors

        if not args.input.is_dir() or args.output.exists() or args.output.resolve().is_relative_to(args.input.resolve()):
            parser.error("Use an existing input package and a new JSONL output outside the input")
        try:
            rows = priors.load_package(args.input, args.family)
            results = priors.evaluate_package(rows, policy=priors.Policy(
                trust_ulf_dateobs=args.trust_ulf_dateobs,
                ulf_duration_convention=args.ulf_duration_convention,
                backend_profile=args.backend_profile))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                for row in results:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({"version": priors.VERSION, "samples": len(results),
                          "confirmed_samples": sum(any(r["confirmed_bits"]) for r in results),
                          "candidate_samples": sum(any(r["candidate_bits"]) for r in results),
                          "backendProfile": args.backend_profile,
                          "backendStrategies": priors.backend_strategy.summarize(
                              [r.get("observations", {}).get("backend_strategy", {}) for r in results]),
                          "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
        return

    if args.command == "explain":
        from .explain import run_explanation

        try:
            result = run_explanation(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "audit":
        from .audit import audit

        result = audit(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "pass" else 1)

    enabled = {item.strip() for item in args.checks.split(",") if item.strip()}
    formal_ids = {f"C{i}" for i in range(41,51)}
    if enabled and enabled <= formal_ids:
        from .formal_checks import run as formal_run
        result = formal_run(args.input,args.output,enabled,json.loads(args.contract.read_text()) if args.contract else None)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    fine_ids = {f"C{i}" for i in range(31,39)}
    if enabled and enabled <= fine_ids:
        from .fine_scan import run as fine_run
        try:
            result = fine_run(args.input, args.output, enabled, args.seed)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    unknown = enabled - set(engine.ALL_CHECKS)
    if unknown or not enabled:
        parser.error(f"Choose one or more known checks; unknown checks: {sorted(unknown)}")
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new directory: {args.output}")
    if args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("Output must be outside the input directory")
    result = engine.run(args.input, args.output, args.seed, enabled, backend_profile=args.backend_profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))
