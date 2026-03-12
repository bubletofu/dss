"""
Visualize ml_pipeline outputs (propensity scores, selections) and plot model efficiency.

Usage:
  python viz_pipeline.py \
    --proposals data/proposals.csv \
    --scores out/proposal_accept_scores.csv \
    --selection out/proposal_selection.csv \
    --outdir out_viz

Outputs:
  - pr_curve.png       (precision-recall)
  - roc_curve.png      (roc)
  - score_hist.png     (accept_prob distribution by label)
  - lift_gain.png      (lift/gain chart)
  - selection_profit.png (expected profit vs cost for selected deals)

Notes:
  - Needs pandas, numpy, matplotlib, scikit-learn (for metrics).
  - Requires true labels: the proposals CSV must contain column "Accepted" (1/0).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score


def load_data(proposals_path: pathlib.Path, scores_path: pathlib.Path, selection_path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    props = pd.read_csv(proposals_path)
    scores = pd.read_csv(scores_path)
    sel = pd.read_csv(selection_path)
    df = scores.merge(props[["ProposalID", "Accepted"]], on="ProposalID", how="left")
    df = df.merge(sel[["ProposalID", "selected", "expected_profit", "Cost"]], on="ProposalID", how="left")
    if df["Accepted"].isna().any():
        raise ValueError("Accepted column missing/NaN for some rows; ensure proposals CSV has Accepted labels.")
    return df, sel


def plot_pr(df: pd.DataFrame, outdir: pathlib.Path):
    y_true = df["Accepted"].values
    scores = df["accept_prob"].values
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=150)
    plt.close()


def plot_roc(df: pd.DataFrame, outdir: pathlib.Path):
    y_true = df["Accepted"].values
    scores = df["accept_prob"].values
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close()


def plot_score_hist(df: pd.DataFrame, outdir: pathlib.Path):
    plt.figure()
    plt.hist(df.loc[df["Accepted"] == 1, "accept_prob"], bins=40, alpha=0.6, label="Accepted", color="tab:blue")
    plt.hist(df.loc[df["Accepted"] == 0, "accept_prob"], bins=40, alpha=0.6, label="Rejected", color="tab:orange")
    plt.xlabel("accept_prob")
    plt.ylabel("Count")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "score_hist.png", dpi=150)
    plt.close()


def plot_lift_gain(df: pd.DataFrame, outdir: pathlib.Path):
    df_sorted = df.sort_values("accept_prob", ascending=False).reset_index(drop=True)
    y = df_sorted["Accepted"].values
    cum_positives = np.cumsum(y)
    total_positives = y.sum()
    perc_customers = (np.arange(1, len(df_sorted) + 1) / len(df_sorted)) * 100
    lift = (cum_positives / (np.arange(1, len(df_sorted) + 1))) / (total_positives / len(df_sorted) + 1e-9)
    gain = (cum_positives / (total_positives + 1e-9)) * 100

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(perc_customers, gain, label="Gain")
    ax[0].plot([0, 100], [0, 100], "k--", alpha=0.5)
    ax[0].set_xlabel("% population (sorted by score)")
    ax[0].set_ylabel("% positives captured")
    ax[0].set_title("Gain Chart")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(perc_customers, lift, label="Lift")
    ax[1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax[1].set_xlabel("% population (sorted by score)")
    ax[1].set_ylabel("Lift")
    ax[1].set_title("Lift Chart")
    ax[1].grid(True, alpha=0.3)

    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(outdir / "lift_gain.png", dpi=150)
    plt.close(fig)


def plot_selection(df: pd.DataFrame, outdir: pathlib.Path):
    sel = df[df["selected"] == 1]
    if sel.empty:
        return
    plt.figure()
    plt.scatter(sel["Cost"], sel["expected_profit"], alpha=0.5, label="Selected deals")
    plt.xlabel("Cost")
    plt.ylabel("Expected Profit")
    plt.title("Selected Deals: Cost vs Expected Profit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "selection_profit.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize ML pipeline outputs")
    ap.add_argument("--proposals", type=pathlib.Path, default=pathlib.Path("data/proposals.csv"))
    ap.add_argument("--scores", type=pathlib.Path, default=pathlib.Path("out/proposal_accept_scores.csv"))
    ap.add_argument("--selection", type=pathlib.Path, default=pathlib.Path("out/proposal_selection.csv"))
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("out_viz"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df, sel = load_data(args.proposals, args.scores, args.selection)

    plot_pr(df, args.outdir)
    plot_roc(df, args.outdir)
    plot_score_hist(df, args.outdir)
    plot_lift_gain(df, args.outdir)
    plot_selection(df, args.outdir)
    print(f"[viz] Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
