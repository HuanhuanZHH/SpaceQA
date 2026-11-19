#!/usr/bin/env python3
"""Plot the audited legacy/enhanced backend smoke comparison.

The values are the six-family, three-sample-per-family read-only audit recorded
in docs/aia94_backend_verification_20261108.md.  They are deliberately labeled
as a smoke audit, not as official precision/recall/F1 results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    ("AIA", 3, 0, 0, 0, 0, 3, 0),
    ("HMI", 3, 0, 0, 0, 0, 3, 0),
    ("CME", 3, 0, 0, 0, 0, 0, 3),
    ("EPB", 3, 0, 0, 0, 0, 0, 3),
    ("DMSP", 3, 1, 1, 0, 0, 1, 2),
    ("ULF", 3, 0, 0, 0, 0, 3, 0),
]


def write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "samples", "legacy_confirmed", "enhanced_confirmed",
                         "legacy_candidates", "enhanced_candidates", "fallback_samples",
                         "native_context_samples"))
        writer.writerows(ROWS)


def plot(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "backend_mode_comparison.csv")
    families = [row[0] for row in ROWS]
    x = np.arange(len(families))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    ax.bar(x - 1.5 * width, [row[2] for row in ROWS], width, label="Legacy confirmed", color="#94a3b8")
    ax.bar(x - 0.5 * width, [row[3] for row in ROWS], width, label="Enhanced confirmed", color="#2563eb")
    ax.bar(x + 0.5 * width, [row[4] for row in ROWS], width, label="Legacy candidates", color="#cbd5e1")
    ax.bar(x + 1.5 * width, [row[5] for row in ROWS], width, label="Enhanced candidates", color="#60a5fa")
    ax.set_title("Legacy vs enhanced backend: audited smoke subset (n=18)")
    ax.set_ylabel("Sample count")
    ax.set_xticks(x, families)
    ax.set_ylim(0, 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    ax.text(0.01, -0.18, "Core findings, confirmed bits and candidate bits: 0 differences across all 18 samples.",
            transform=ax.transAxes, fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(out / "backend_mode_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    fallback = np.array([row[6] for row in ROWS])
    native = np.array([row[7] for row in ROWS])
    ax.bar(x, native, label="Context available", color="#16a34a")
    ax.bar(x, fallback, bottom=native, label="Bounded fallback / needs source", color="#f59e0b")
    ax.set_title("Enhanced backend observation coverage (same audited subset)")
    ax.set_ylabel("Sample count")
    ax.set_xticks(x, families)
    ax.set_ylim(0, 3.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.01, -0.18, "Fallback is an explicit safety state; it does not add or remove a defect verdict.",
            transform=ax.transAxes, fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(out / "backend_observation_coverage.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/backend_comparison_20261108"))
    args = parser.parse_args()
    plot(args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
