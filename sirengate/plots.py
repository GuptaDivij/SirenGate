from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_tradeoff(summary_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    label_col = "policy" if "policy" in df.columns else "threshold"

    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        plt.scatter(row["avg_bytes_per_clip"], row["macro_f1"], s=80)
        plt.annotate(
            row[label_col],
            (row["avg_bytes_per_clip"], row["macro_f1"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Average bytes transmitted per clip")
    plt.ylabel("Macro F1")
    plt.title("Bandwidth vs detection performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_trace(events_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(events_csv)

    plt.figure(figsize=(9, 4))
    plt.plot(df["threshold_used"].values)
    plt.xlabel("Streaming step")
    plt.ylabel("Threshold")
    plt.title("Adaptive threshold over stream")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
