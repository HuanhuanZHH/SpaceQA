"""Create isolated replay copies; refuse an existing workspace or input overwrite.

Only known absolute path string constants change. Archived scientific sources
remain byte-identical. Generated source hashes and every changed literal are
recorded in a receipt. This command stages code; it starts no experiment.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parent
OLD_OPC = "/staff/tangjiarui/OPC"
OLD_REPO = OLD_OPC + "/opc-agent"
OLD_RUN = OLD_OPC + "/inference_prior_20261102"
SOURCES = (
    "prior_rules.py", "compare_versions.py", "compare_gbdt_extension.py",
    "compare_headwise.py", "evaluate_roadshow_prior.py", "run_roadshow_prior.py",
    "fuse_roadshow_extension.py", "fuse_headwise_roadshow.py",
    "gbdt/train_gbdt.py", "gbdt/predict_roadshow.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def change_path(value, replacements):
    if not isinstance(value, str):
        return value
    for before, after in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
        if value == before or value.startswith(before + "/"):
            return after + value[len(before):]
    return value


def relocate_source(source, replacements):
    """Return code/receipt; all non-path AST nodes are mechanically preserved."""
    original = ast.parse(source)
    expected = copy.deepcopy(original)
    changes = []
    for node in ast.walk(expected):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            updated = change_path(node.value, replacements)
            if updated != node.value:
                changes.append(dict(line=node.lineno, before=node.value, after=updated))
                node.value = updated
    generated = ast.unparse(expected) + "\n"
    reparsed = ast.parse(generated)
    if ast.dump(reparsed, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Generated code changed beyond the declared path literals")
    # Restore just the changed literal nodes and compare with the original AST.
    restored = copy.deepcopy(expected)
    changed_by_line = {(c["line"], c["after"]): c["before"] for c in changes}
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = changed_by_line.get((node.lineno, node.value), node.value)
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("AST preservation check failed")
    return generated, changes


def stage(workspace, *, opc_root=OLD_OPC, repo=OLD_REPO, archive=ARCHIVE):
    workspace, archive = Path(workspace).resolve(), Path(archive).resolve()
    if workspace.exists():
        raise FileExistsError("Choose a new workspace; existing runs are never overwritten")
    if workspace == archive or workspace.is_relative_to(archive):
        raise ValueError("Replay workspace must be outside the archived experiment")
    replacements = {
        OLD_RUN: (workspace / "runs").as_posix(),
        OLD_REPO: Path(repo).as_posix().rstrip("/"),
        OLD_OPC: Path(opc_root).as_posix().rstrip("/"),
    }
    generated = []
    for name in SOURCES:
        raw = (archive / name).read_bytes()
        text, changes = relocate_source(raw.decode("utf-8-sig"), replacements)
        generated.append((name, raw, text.encode("utf-8"), changes))
    (workspace / "code").mkdir(parents=True)
    (workspace / "runs").mkdir()
    receipts = []
    for name, raw, data, changes in generated:
        target = workspace / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        receipts.append(dict(path=name, archived_sha256=digest(raw), generated_sha256=digest(data), changed_literals=changes))
    report = dict(status="staged", workspace=str(workspace), output_root=str(workspace / "runs"),
                  source_archive=str(archive), path_replacements=replacements, files=receipts,
                  scientific_AST_changes="none except listed path string constants", experiments_started=False,
                  archive_source_modified=False, test_or_training_data_read=False)
    (workspace / "stage_receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path, help="New directory outside this archive; must not exist")
    parser.add_argument("--opc-root", default=OLD_OPC, help="Read-only external model/data root")
    parser.add_argument("--repo", default=OLD_REPO, help="Read-only historical repository/cache root")
    args = parser.parse_args()
    report = stage(args.workspace, opc_root=args.opc_root, repo=args.repo)
    print(json.dumps(dict(status=report["status"], workspace=report["workspace"], files=len(report["files"]), experiments_started=False)))


if __name__ == "__main__":
    main()
