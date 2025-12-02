"""
End-to-end DSS pipeline:
- ETL: load/clean proposal data
- Train classifier for rejection propensity (accept_prob = 1 - reject_prob)
- Train K-Means for customer segmentation
- Optimize proposal selection via knapsack (budget constraint)
- Export CSVs for Power BI and a plain-text report

Input CSV expected columns (min):
  ProposalID, ProposalAmount, Cost, Region, CustomerType, ProductCategory, Accepted
Optional behavior/segmentation columns:
  Purchases, AvgOrderValue, PromoResponseRate, Recency

Usage:
  python3 ml_pipeline.py --data data/proposals.csv --budget 500000
Outputs go to ./out by default.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier


@dataclass
class TrainResult:
    auc: float
    class_report: str
    pr_auc: float
    best_threshold: float
    best_f1: float
    positive_label_name: str
    segment_k: Optional[int]
    segment_silhouette: Optional[float]
    total_selected_cost: float
    total_selected_profit: float


def load_data(path: pathlib.Path) -> pd.DataFrame:
    print(f"[ETL] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"[ETL] Loaded {len(df):,} rows with columns: {list(df.columns)}")

    def coerce_numeric(series: pd.Series, floor_zero: bool = False, name: str = "") -> pd.Series:
        s = series.astype(str).str.strip()
        s = s.str.replace(r"[^\d,.\-]", "", regex=True)
        comma_decimal_mask = s.str.contains(",") & ~s.str.contains(r"\.")
        s.loc[comma_decimal_mask] = s.loc[comma_decimal_mask].str.replace(",", ".")
        s = s.str.replace(",", "", regex=False)
        cleaned = pd.to_numeric(s, errors="coerce")
        if floor_zero:
            cleaned = cleaned.mask(cleaned < 0, np.nan)
        if name:
            missing = cleaned.isna().sum()
            if missing:
                print(f"[ETL] {name}: coerced to numeric with {missing} missing/invalid")
        return cleaned

    if "ProposalAmount" in df.columns:
        df["ProposalAmount"] = coerce_numeric(df["ProposalAmount"], floor_zero=True, name="ProposalAmount")
    if "Cost" in df.columns:
        df["Cost"] = coerce_numeric(df["Cost"], floor_zero=True, name="Cost")

    if {"ProposalAmount", "Cost"}.issubset(df.columns):
        df["margin"] = df["ProposalAmount"] - df["Cost"]
        df["margin_rate"] = df["margin"] / df["ProposalAmount"].replace({0: np.nan})
        df["margin_rate"] = df["margin_rate"].replace([np.inf, -np.inf], np.nan)
        df["cost_rate"] = df["Cost"] / df["ProposalAmount"].replace({0: np.nan})
        df["cost_rate"] = df["cost_rate"].replace([np.inf, -np.inf], np.nan)
        df["log_amount"] = np.log1p(df["ProposalAmount"])
        df["log_cost"] = np.log1p(df["Cost"])

    for date_col in ["ProposalDate", "OrderDate", "Date"]:
        if date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                df["month"] = dates.dt.month
                df["quarter"] = dates.dt.quarter
                break
            except Exception:
                pass

    # Optional: discount depth if available
    if "DiscountRate" in df.columns:
        df["discount_rate"] = coerce_numeric(df["DiscountRate"], floor_zero=True, name="DiscountRate")
    elif "UnitPriceDiscount" in df.columns:
        df["discount_rate"] = coerce_numeric(df["UnitPriceDiscount"], floor_zero=True, name="UnitPriceDiscount")
    return df


def train_logreg(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, TrainResult]:
    """
    Train a rejection-propensity classifier using LightGBM on a balanced sample,
    then score the full dataset and return model + metrics.

    Positive class = Rejected (Accepted == 0).
    """
    required_cols = ["ProposalID", "ProposalAmount", "Region", "CustomerType", "ProductCategory", "Accepted"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for classifier: {missing}")

    # ---- 1) Robust feature engineering & clipping ---------------------------------
    for col in ["ProposalAmount", "Cost", "margin", "margin_rate", "cost_rate", "log_amount", "log_cost"]:
        if col in df.columns:
            low, high = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=low, upper=high)

    # Rich feature set (use whatever exists in this dataset)
    feature_cols = [
        "ProposalAmount",
        "Cost",
        "margin",
        "margin_rate",
        "cost_rate",
        "log_amount",
        "log_cost",
        "discount_rate",
        "month",
        "quarter",
        "Purchases",
        "AvgOrderValue",
        "PromoResponseRate",
        "Recency",
        "Region",
        "CustomerType",
        "ProductCategory",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    # Positive class = Rejected
    y = (df["Accepted"] == 0).astype(int)

    cat_cols = [c for c in ["Region", "CustomerType", "ProductCategory"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ],
        sparse_threshold=0.0,
    )

    # ---- 2) Balanced sampling: giữ tất cả Rejected + sample Accepted ----------------
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()

    if len(pos_idx) == 0:
        raise ValueError("No rejected samples found; cannot train rejection model.")

    rng = np.random.RandomState(42)
    neg_sample_size = min(len(neg_idx), len(pos_idx) * 5)  # ratio ~ 1:5
    neg_sample_idx = rng.choice(neg_idx, neg_sample_size, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_sample_idx])
    rng.shuffle(sample_idx)

    max_samples = 30000
    if len(sample_idx) > max_samples:
        sample_idx = sample_idx[:max_samples]

    X_sample, y_sample = X.loc[sample_idx], y.loc[sample_idx]

    Xtr, Xte, ytr, yte = train_test_split(
        X_sample,
        y_sample,
        test_size=0.2,
        stratify=y_sample,
        random_state=42,
    )

    # ---- 3) Preprocess --------------------------------------------------------------
    Xtr_prep = pre.fit_transform(Xtr)
    Xte_prep = pre.transform(Xte)

    # Optional oversampling flag lấy từ df (
    oversample = getattr(df, "_oversample", False) if hasattr(df, "_oversample") else False
    if oversample:
        try:
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=42)
            Xtr_prep, ytr = ros.fit_resample(Xtr_prep, ytr)
            print(f"[LightGBM] Oversampled training set to {len(ytr)} rows")
        except Exception:
            raise RuntimeError(
                "Oversampling requested but 'imbalanced-learn' is not installed. "
                "Install with 'pip install imbalanced-learn'"
            )

    # ---- 4) LightGBM classifier (tuned) --------------------------------------------
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced", 
        n_jobs=-1,
        random_state=42,
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=ytr)

    print(f"[LightGBM] Training on {Xtr_prep.shape[0]:,} samples (preprocessed)...")
    clf.fit(
        Xtr_prep,
        ytr,
        sample_weight=sample_weights,
        eval_set=[(Xte_prep, yte)],
        eval_metric="auc",
    )

    # ---- 5) Evaluation trên validation set -----------------------------------------
    preds = clf.predict_proba(Xte_prep)[:, 1]  # P(Rejected)
    auc = roc_auc_score(yte, preds)
    pr_auc = average_precision_score(yte, preds)

    precision, recall, thresholds = precision_recall_curve(yte, preds)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores[:-1].argmax()  # bỏ điểm cuối của curve
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    report = classification_report(
        yte,
        (preds >= best_threshold).astype(int),
        digits=3,
        target_names=["Accepted", "Rejected"],
        zero_division=0,
    )

    print(
        f"[LightGBM] (Positive=Rejected) "
        f"AUC={auc:.3f} | PR-AUC={pr_auc:.3f} | "
        f"best_thr={best_threshold:.3f} | best_f1={best_f1:.3f}"
    )

    # ---- 6) Score full dataset ------------------------------------------------------
    df_scored = df.copy()
    X_prep = pre.transform(X)
    reject_prob = clf.predict_proba(X_prep)[:, 1]
    df_scored["accept_prob"] = 1 - reject_prob

    final_model = Pipeline(steps=[("prep", pre), ("clf", clf)])

    return final_model, df_scored, TrainResult(
        auc=auc,
        class_report=report,
        pr_auc=pr_auc,
        best_threshold=best_threshold,
        best_f1=best_f1,
        positive_label_name="Rejected",
        segment_k=None,
        segment_silhouette=None,
        total_selected_cost=0.0,
        total_selected_profit=0.0,
    )


def train_kmeans(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Optional[int], Optional[float]]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[KMeans] Skipping segmentation, missing cols: {missing}")
        return df, None, None

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    if X.dropna(how="all").empty:
        print("[KMeans] No usable rows; skipping segmentation.")
        return df, None, None

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    variances = np.var(X_imputed, axis=0)
    keep_mask = variances > 1e-9
    if not keep_mask.any():
        print("[KMeans] All segmentation features have zero variance; skipping segmentation.")
        return df, None, None

    X_use = X_imputed[:, keep_mask]
    max_kmeans_rows = 20000
    if X_use.shape[0] > max_kmeans_rows:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(X_use.shape[0], max_kmeans_rows, replace=False)
        X_use = X_use[sample_idx]
        original_idx = X.index.to_numpy()[sample_idx]
    else:
        original_idx = X.index.to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_use)

    if Xs.shape[0] < 10:
        print(f"[KMeans] Too few rows ({Xs.shape[0]}) for clustering; skipping segmentation.")
        return df, None, None

    best_k, best_s, best_model = None, -1, None
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        if s > best_s:
            best_k, best_s, best_model = k, s, km

    print(f"[KMeans] Best k={best_k} silhouette={best_s:.3f}")
    labels = best_model.predict(Xs)
    df_seg = df.copy()
    df_seg["segment"] = np.nan
    df_seg.loc[original_idx, "segment"] = labels
    return df_seg, best_k, best_s


def optimize_knapsack(df: pd.DataFrame, budget: float, top_n: int = 5000) -> Tuple[pd.DataFrame, float, float]:
    required = ["ProposalID", "ProposalAmount", "Cost", "accept_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for optimization: {missing}")

    df = df.copy()
    df["expected_revenue"] = df["accept_prob"] * df["ProposalAmount"]
    df["expected_profit"] = df["expected_revenue"] - df["Cost"]
    df = df[df["expected_profit"] > 0].reset_index(drop=True)
    if df.empty:
        print("[Opt] No positive expected profit proposals; skipping selection.")
        df["selected"] = 0
        return df, 0.0, 0.0

    # Reduce problem size for tractable ILP solve
    df["profit_per_cost"] = df["expected_profit"] / df["Cost"].replace({0: np.nan})
    df["profit_per_cost"] = df["profit_per_cost"].replace([np.inf, -np.inf], 0)
    if len(df) > top_n:
        df = df.nlargest(top_n, ["profit_per_cost", "expected_profit"]).reset_index(drop=True)
        print(f"[Opt] Reduced to top {top_n} proposals by profit efficiency.")

    prob = LpProblem("proposal_knapsack", LpMaximize)
    x = {r.ProposalID: LpVariable(f"x_{r.ProposalID}", 0, 1, LpBinary) for r in df.itertuples()}

    prob += lpSum(x[r.ProposalID] * r.expected_profit for r in df.itertuples())
    prob += lpSum(x[r.ProposalID] * r.Cost for r in df.itertuples()) <= budget

    prob.solve(PULP_CBC_CMD(msg=False))

    df["selected"] = df["ProposalID"].apply(lambda pid: int(x[pid].value()))
    total_profit = float((df["selected"] * df["expected_profit"]).sum())
    total_cost = float((df["selected"] * df["Cost"]).sum())
    print(f"[Opt] Selected {df['selected'].sum()} proposals; cost={total_cost:,.0f}, profit={total_profit:,.0f}")
    return df, total_cost, total_profit


def save_outputs(out_dir: pathlib.Path, df_scored: pd.DataFrame, df_seg: pd.DataFrame, df_opt: pd.DataFrame, model: Pipeline, report: TrainResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal_accept_scores.csv").write_text(df_scored.to_csv(index=False))
    (out_dir / "customer_segments.csv").write_text(df_seg.to_csv(index=False))
    (out_dir / "proposal_selection.csv").write_text(df_opt.to_csv(index=False))
    joblib.dump(model, out_dir / "logreg_proposal.joblib")

    report_path = out_dir / "model_report.txt"
    with report_path.open("w") as f:
        f.write("=== Classifier (positive = Rejected) ===\n")
        f.write(f"AUC: {report.auc:.4f}\n")
        f.write(f"PR-AUC: {report.pr_auc:.4f}\n")
        f.write(f"Best threshold (F1): {report.best_threshold:.4f}, Best F1: {report.best_f1:.4f}\n\n")
        f.write(report.class_report)
        f.write("\n\n=== K-Means ===\n")
        if report.segment_k:
            f.write(f"k: {report.segment_k}, silhouette: {report.segment_silhouette:.4f}\n")
        else:
            f.write("Segmentation skipped or insufficient data\n")
        f.write("\n=== Optimization ===\n")
        f.write(f"Total selected cost: {report.total_selected_cost:,.2f}\n")
        f.write(f"Total selected expected profit: {report.total_selected_profit:,.2f}\n")
    print(f"[IO] Saved outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSS pipeline: propensity + segmentation + optimization")
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to proposals CSV")
    parser.add_argument("--budget", type=float, default=500000.0, help="Budget constraint for optimization")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"), help="Output directory")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip customer segmentation step")
    parser.add_argument("--opt-top-n", type=int, default=5000, help="Cap proposals used in optimization (ILP) for tractability")
    parser.add_argument("--model", choices=["rf", "lightgbm"], default="rf", help="Model to use for classification: 'rf' or 'lightgbm'")
    parser.add_argument("--oversample", action="store_true", help="Apply RandomOverSampler to training data to address class imbalance")
    return parser.parse_args()


def main():
    args = parse_args()
    df_raw = load_data(args.data)

    df_raw._model_choice = args.model
    df_raw._oversample = args.oversample

    model, df_scored, base_report = train_logreg(df_raw)

    # K-Means segmentation (optional columns)
    seg_features = ["Purchases", "AvgOrderValue", "PromoResponseRate", "Recency"]
    if args.skip_kmeans:
        print("[KMeans] Skipped by flag.")
        df_seg, k, sil = df_scored, None, None
    else:
        df_seg, k, sil = train_kmeans(df_scored, seg_features)

    df_opt, total_cost, total_profit = optimize_knapsack(df_seg, budget=args.budget, top_n=args.opt_top_n)

    report = TrainResult(
        auc=base_report.auc,
        class_report=base_report.class_report,
        pr_auc=base_report.pr_auc,
        best_threshold=base_report.best_threshold,
        best_f1=base_report.best_f1,
        positive_label_name=base_report.positive_label_name,
        segment_k=k,
        segment_silhouette=sil,
        total_selected_cost=total_cost,
        total_selected_profit=total_profit,
    )

    save_outputs(args.out, df_scored, df_seg, df_opt, model, report)


if __name__ == "__main__":
    main()