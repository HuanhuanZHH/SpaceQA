"""Run only fresh roadshow prior inspection, then stop at prior_complete.json.

This safe entry preserves the archived runner and excludes its obsolete model
fusion tail. Use fuse_roadshow_extension.py or fuse_headwise_roadshow.py later.
Inputs stay read-only, and --output must be a new child of the staged run root.
--frozen-policy is retained for CLI compatibility; this fresh-only phase does
not read it or tune routes on roadshow data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def fresh_only_tree(tree):
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    if main is None:
        raise ValueError("Archived main function missing")
    checkpoints = []
    for index, statement in enumerate(main.body):
        expression = statement.value if isinstance(statement, ast.Expr) else None
        if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "event"
                and expression.args and isinstance(expression.args[0], ast.Constant)
                and expression.args[0].value == "roadshow_prior_complete"):
            checkpoints.append(index)
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one completed fresh-prior checkpoint")
    main.body = main.body[:checkpoints[0] + 1]
    return tree


def main():
    source = Path(__file__).with_name("evaluate_roadshow_prior.py")
    raw = source.read_bytes()
    tree = fresh_only_tree(ast.parse(raw.decode("utf-8-sig"), filename=str(source)))
    namespace = dict(__name__="archived_fresh_prior_only", __file__=str(source))
    exec(compile(tree, str(source), "exec"), namespace)
    namespace["__doc__"] = __doc__
    namespace["main"]()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    receipt = dict(status="fresh_prior_complete", archived_runner_sha256=hashlib.sha256(raw).hexdigest(),
                   entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   transformation="main ends immediately after the roadshow_prior_complete checkpoint",
                   obsolete_fusion_executed=False, B_training_or_threshold_selection=False)
    (args.output / "fresh_entry_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
