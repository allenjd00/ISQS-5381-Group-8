import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PIPELINE_STEPS = [
    "01_data_audit.py",
    "04_flatten_longformat_ef_gr.py",
    "02_build_analysis_base.py",
    "05_build_project_target.py",
    "03_build_modeling_ready.py",
    "06_train_baseline.py",
]


def run_step(step_file: str) -> dict:
    step_path = SCRIPTS / step_file
    started_at = datetime.now(timezone.utc).isoformat()
    cmd = [sys.executable, str(step_path)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ended_at = datetime.now(timezone.utc).isoformat()

    return {
        "step": step_file,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "return_code": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def safe_shape(csv_path: Path):
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path, low_memory=False)
    return {"rows": int(len(frame)), "cols": int(frame.shape[1])}


step_results = [run_step(step) for step in PIPELINE_STEPS]
failed_steps = [step for step in step_results if step["return_code"] != 0]

run_manifest = {
    "run_started_utc": step_results[0]["started_at_utc"],
    "run_ended_utc": step_results[-1]["ended_at_utc"],
    "python_executable": sys.executable,
    "pipeline_steps": step_results,
    "status": "failed" if failed_steps else "success",
    "output_shapes": {
        "data_audit_summary": safe_shape(OUT / "data_audit_summary.csv"),
        "input_inventory": safe_shape(OUT / "input_inventory.csv"),
        "analysis_base": safe_shape(OUT / "analysis_base.csv"),
        "project_target_unitid": safe_shape(OUT / "project_target_unitid.csv"),
        "modeling_ready": safe_shape(OUT / "modeling_ready.csv"),
        "variable_map": safe_shape(OUT / "variable_map.csv"),
        "baseline_feature_importance": safe_shape(OUT / "baseline_feature_importance.csv"),
    },
}

manifest_path = OUT / "pipeline_run_manifest.json"
with manifest_path.open("w", encoding="utf-8") as fp:
    json.dump(run_manifest, fp, indent=2)

if failed_steps:
    print("Pipeline failed. See manifest for details:")
    print(f"- {manifest_path}")
    for step in failed_steps:
        print(f"FAILED: {step['step']} (return_code={step['return_code']})")
    raise SystemExit(1)

print("Pipeline completed successfully.")
print(f"- {manifest_path}")
