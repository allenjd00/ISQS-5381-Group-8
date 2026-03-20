import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

csv_files = sorted(ROOT.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in workspace root.")

summary_rows = []
dfs = {}
inventory_rows = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    dfs[f.stem] = df

    row = {
        "file": f.name,
        "rows": len(df),
        "cols": df.shape[1],
        "has_UNITID": "UNITID" in df.columns,
        "unitid_nonnull": int(df["UNITID"].notna().sum()) if "UNITID" in df.columns else 0,
        "unitid_unique": int(df["UNITID"].nunique(dropna=True)) if "UNITID" in df.columns else 0,
        "missing_pct_mean": float(df.isna().mean().mean()),
    }
    summary_rows.append(row)

    inventory_rows.append(
        {
            "file": f.name,
            "size_bytes": int(f.stat().st_size),
            "sha256": sha256_file(f),
            "rows": len(df),
            "cols": df.shape[1],
            "has_UNITID": "UNITID" in df.columns,
            "one_row_per_UNITID": bool("UNITID" in df.columns and len(df) == df["UNITID"].nunique(dropna=True)),
        }
    )

summary = pd.DataFrame(summary_rows).sort_values("file")
summary.to_csv(OUT / "data_audit_summary.csv", index=False)

unitid_sets = {}
for name, df in dfs.items():
    if "UNITID" in df.columns:
        unitid_sets[name] = set(df["UNITID"].dropna().astype(str).unique())

coverage = {}
if unitid_sets:
    all_names = list(unitid_sets.keys())
    intersection = set.intersection(*[unitid_sets[n] for n in all_names])
    union = set.union(*[unitid_sets[n] for n in all_names])
    coverage["files_with_UNITID"] = all_names
    coverage["intersection_count"] = len(intersection)
    coverage["union_count"] = len(union)
    coverage["intersection_rate"] = len(intersection) / len(union) if union else 0.0

    pairwise = {}
    for i, a in enumerate(all_names):
        for b in all_names[i + 1 :]:
            inter = unitid_sets[a] & unitid_sets[b]
            denom = unitid_sets[a] | unitid_sets[b]
            pairwise[f"{a}__{b}"] = {
                "intersection": len(inter),
                "jaccard": len(inter) / len(denom) if denom else 0.0,
            }
    coverage["pairwise"] = pairwise

with open(OUT / "unitid_join_coverage.json", "w", encoding="utf-8") as fp:
    json.dump(coverage, fp, indent=2)

inventory_df = pd.DataFrame(inventory_rows).sort_values("file")
inventory_df.to_csv(OUT / "input_inventory.csv", index=False)

print("Wrote:")
print(f"- {OUT / 'data_audit_summary.csv'}")
print(f"- {OUT / 'unitid_join_coverage.json'}")
print(f"- {OUT / 'input_inventory.csv'}")
