import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")
    return pd.read_csv(path, low_memory=False)


def sanitize_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    return clean.replace({-3: np.nan, -2: np.nan, -1: np.nan, "-3": np.nan, "-2": np.nan, "-1": np.nan})


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {col: f"{prefix}_{col}" for col in df.columns if col != "UNITID"}
    return df.rename(columns=rename_map)


def load_metadata(xlsx_path: Path, source_key: str) -> pd.DataFrame:
    if not xlsx_path.exists():
        return pd.DataFrame(columns=["varName", "DataType", "varTitle", "longDescription", "source"])

    varlist = pd.read_excel(xlsx_path, sheet_name="Varlist")
    desc = pd.read_excel(xlsx_path, sheet_name="Description")

    varlist = varlist[["varName", "DataType", "varTitle"]].copy()
    desc = desc[["varName", "longDescription"]].copy()

    md = varlist.merge(desc, on="varName", how="left")
    md["source"] = source_key
    return md


def identify_code_like_features(feature_cols: list[str], metadata: pd.DataFrame) -> list[str]:
    # Keep policy-relevant institutional indicators even when dictionary text says "code to indicate".
    policy_indicator_allowlist = {
        "hd_HBCU",
        "hd_TRIBAL",
        "hd_MEDICAL",
        "hd_HOSPITAL",
        "hd_LANDGRNT",
    }

    if metadata.empty:
        metadata_lookup = {}
    else:
        metadata_lookup = {
            f"{row['source']}_{row['varName']}": {
                "title": str(row.get("varTitle", "") or "").lower(),
                "desc": str(row.get("longDescription", "") or "").lower(),
            }
            for _, row in metadata.iterrows()
        }

    text_keywords = [
        " code",
        "identifier",
        "allocation factor",
        "fips",
        "zip code",
        "index",
    ]
    name_pattern = re.compile(r"(_idx_|_idx$|_id$|_id\d*$|_ein$|_cod$|_code$|_fips$|_zip$|_cbsa$|_csa$)", re.IGNORECASE)

    excluded = []
    for feature in feature_cols:
        if feature in policy_indicator_allowlist:
            continue

        if name_pattern.search(feature):
            excluded.append(feature)
            continue

        md = metadata_lookup.get(feature)
        if not md:
            continue

        text = f" {md['title']} {md['desc']} "
        if any(keyword in text for keyword in text_keywords):
            excluded.append(feature)

    return sorted(set(excluded))


def build_target_from_gr(gr: pd.DataFrame) -> pd.DataFrame:
    required = ["UNITID", "CHRTSTAT", "GRTOTLT"]
    missing = [col for col in required if col not in gr.columns]
    if missing:
        raise ValueError(f"gr2024.csv missing required columns: {missing}")

    work = sanitize_sentinels(gr)
    work["CHRTSTAT"] = pd.to_numeric(work["CHRTSTAT"], errors="coerce")
    work["GRTOTLT"] = pd.to_numeric(work["GRTOTLT"], errors="coerce")

    adjusted = (
        work.loc[work["CHRTSTAT"] == 12]
        .groupby("UNITID", dropna=False)["GRTOTLT"]
        .sum(min_count=1)
        .rename("gr_adjusted_cohort_total")
    )
    completers_150 = (
        work.loc[work["CHRTSTAT"] == 13]
        .groupby("UNITID", dropna=False)["GRTOTLT"]
        .sum(min_count=1)
        .rename("gr_completers_150_total")
    )

    target = pd.concat([adjusted, completers_150], axis=1).reset_index()
    target["gr_completion_rate_150"] = np.where(
        target["gr_adjusted_cohort_total"] > 0,
        target["gr_completers_150_total"] / target["gr_adjusted_cohort_total"],
        np.nan,
    )

    valid_rate = target["gr_completion_rate_150"].dropna()
    if valid_rate.empty:
        raise ValueError("No valid gr_completion_rate_150 values could be calculated")

    median_cut = float(valid_rate.median())
    target["y_grad_outcome_high"] = np.where(
        target["gr_completion_rate_150"].notna(),
        (target["gr_completion_rate_150"] >= median_cut).astype(int),
        np.nan,
    )

    return target


def flatten_ef_minimal(ef: pd.DataFrame) -> pd.DataFrame:
    if "UNITID" not in ef.columns:
        raise ValueError("ef2024a.csv missing required column: UNITID")

    work = sanitize_sentinels(ef)
    grouped = work.groupby("UNITID", dropna=False)
    out = pd.DataFrame({"UNITID": grouped.size().index})
    out["ef_rows"] = grouped.size().values

    if "EFTOTLT" in work.columns:
        work["EFTOTLT"] = pd.to_numeric(work["EFTOTLT"], errors="coerce")
        out["ef_eftotlt_max"] = grouped["EFTOTLT"].max().values
        out["ef_eftotlt_mean"] = grouped["EFTOTLT"].mean().values

    return out


def main() -> None:
    source_files = {
        "hd": ROOT / "hd2024.csv",
        "ic": ROOT / "ic2024.csv",
        "flags": ROOT / "flags2024.csv",
        "adm": ROOT / "adm2024.csv",
        "sfa": ROOT / "sfa2324.csv",
        "cost1": ROOT / "cost1_2024.csv",
        "cost2": ROOT / "cost2_2024.csv",
        "ef": ROOT / "ef2024a.csv",
        "gr": ROOT / "gr2024.csv",
    }

    loaded = {name: load_csv(path) for name, path in source_files.items()}
    loaded = {name: sanitize_sentinels(df) for name, df in loaded.items()}

    metadata_files = [
        (ROOT / "hd2024.xlsx", "hd"),
        (ROOT / "ic2024.xlsx", "ic"),
        (ROOT / "flags2024.xlsx", "flags"),
        (ROOT / "adm2024.xlsx", "adm"),
        (ROOT / "sfa2324.xlsx", "sfa"),
        (ROOT / "COST1_2024.xlsx", "cost1"),
        (ROOT / "cost1_2024.xlsx", "cost1"),
        (ROOT / "cost2_2024.xlsx", "cost2"),
    ]

    metadata_frames = []
    loaded_sources = set()
    for path, source in metadata_files:
        if source in loaded_sources and source == "cost1":
            continue
        md = load_metadata(path, source)
        if not md.empty:
            metadata_frames.append(md)
            loaded_sources.add(source)
    metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()

    direct_sources = {name: loaded[name] for name in ["hd", "ic", "flags", "adm", "sfa", "cost1", "cost2"]}
    target = build_target_from_gr(loaded["gr"])
    ef_flat = flatten_ef_minimal(loaded["ef"])

    for name, df in direct_sources.items():
        if "UNITID" not in df.columns:
            raise ValueError(f"{name} is missing required key column UNITID")
    if "UNITID" not in target.columns or "UNITID" not in ef_flat.columns:
        raise ValueError("Derived tables must include UNITID")

    prefixed = {name: prefix_columns(df, name) for name, df in direct_sources.items()}
    ef_prefixed = prefix_columns(ef_flat, "ef")

    # Build a wide institution table; GR is used only for target construction to avoid leakage.
    wide = prefixed["hd"].copy()
    for name in ["ic", "flags", "adm", "sfa", "cost1", "cost2"]:
        wide = wide.merge(prefixed[name], on="UNITID", how="left")
    wide = wide.merge(ef_prefixed, on="UNITID", how="left")
    wide = wide.merge(
        target[["UNITID", "gr_completion_rate_150", "y_grad_outcome_high"]],
        on="UNITID",
        how="left",
    )

    model_df = wide.dropna(subset=["y_grad_outcome_high"]).copy()
    model_df["y_grad_outcome_high"] = pd.to_numeric(model_df["y_grad_outcome_high"], errors="coerce")
    model_df = model_df.dropna(subset=["y_grad_outcome_high"]).copy()
    model_df["y_grad_outcome_high"] = model_df["y_grad_outcome_high"].astype(int)

    drop_cols = {"UNITID", "y_grad_outcome_high", "gr_completion_rate_150"}
    raw_feature_cols = [col for col in model_df.columns if col not in drop_cols]
    excluded_code_like = identify_code_like_features(raw_feature_cols, metadata)
    filtered_feature_cols = [col for col in raw_feature_cols if col not in set(excluded_code_like)]

    # Basic exploration uses numeric-coercible features only.
    X_numeric = model_df[filtered_feature_cols].apply(pd.to_numeric, errors="coerce")
    y = model_df["y_grad_outcome_high"]

    min_non_null_ratio = 0.05
    coverage = X_numeric.notna().mean()
    keep_coverage = coverage[coverage >= min_non_null_ratio].index.tolist()
    X_numeric = X_numeric[keep_coverage]

    if X_numeric.empty:
        raise ValueError("No usable numeric features after non-null coverage filtering")

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index)

    nunique = X_imputed.nunique(dropna=False)
    keep_non_constant = nunique[nunique > 1].index.tolist()
    X_final = X_imputed[keep_non_constant]

    if X_final.empty:
        raise ValueError("No usable features after constant-column filtering")

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    impurity = pd.DataFrame(
        {
            "feature": X_final.columns,
            "rf_impurity_importance": model.feature_importances_,
        }
    )

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    permutation = pd.DataFrame(
        {
            "feature": X_final.columns,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )

    importance = impurity.merge(permutation, on="feature", how="left")
    importance = importance.sort_values(
        ["permutation_importance_mean", "rf_impurity_importance"],
        ascending=False,
    )

    wide_path = OUT / "analysis_fulljoin_explore.csv"
    importance_path = OUT / "feature_importance_fulljoin_explore.csv"
    report_path = OUT / "feature_importance_fulljoin_explore_report.json"

    wide.to_csv(wide_path, index=False)
    importance.to_csv(importance_path, index=False)

    report = {
        "source_files": [path.name for path in source_files.values()],
        "notes": {
            "target_policy": "gr2024 used only to construct target to avoid direct leakage from raw graduation rows",
            "feature_policy": "code-like fields removed via data dictionary + name heuristics; then numeric-coercible columns with >=5% non-null coverage, median-imputed, non-constant only",
        },
        "row_counts": {
            "wide_rows": int(len(wide)),
            "model_rows": int(len(model_df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "feature_counts": {
            "raw_features": int(len(raw_feature_cols)),
            "excluded_code_like": int(len(excluded_code_like)),
            "after_code_filter": int(len(filtered_feature_cols)),
            "after_coverage_filter": int(len(keep_coverage)),
            "after_non_constant_filter": int(len(keep_non_constant)),
        },
        "excluded_code_like_features": excluded_code_like,
        "model_metrics": metrics,
        "top_20_permutation_features": importance.head(20).to_dict(orient="records"),
    }

    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("Wrote:")
    print(f"- {wide_path}")
    print(f"- {importance_path}")
    print(f"- {report_path}")
    print(f"Rows (wide/model/train/test): {len(wide)}/{len(model_df)}/{len(X_train)}/{len(X_test)}")
    print(
        "Feature counts (raw/code-excluded/coverage/non-constant): "
        f"{len(raw_feature_cols)}/{len(filtered_feature_cols)}/{len(keep_coverage)}/{len(keep_non_constant)}"
    )
    print(f"Metrics: accuracy={metrics['accuracy']:.4f}, roc_auc={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
