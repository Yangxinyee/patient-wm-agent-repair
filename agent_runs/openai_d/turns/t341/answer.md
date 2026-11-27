The final cohort preserved all four response directions. Insulin’s shifts were somewhat smaller; norepinephrine’s were consistent with earlier cohorts. I’m retaining the validated checkpoint: another untested edit would risk factual fit.

I’ll pool all 400 stays and verify the saved weights before submission. Consistent pooled responses and an exact checkpoint match will confirm this choice; a mismatch would indicate an integrity problem.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

expected_edits = {
    "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias",
    "attr_heads.value_bin.2.weight",
    "embed_transition.value_bin.weight",
}
assert set(edited()) == expected_edits

def as_array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

with np.load("/work/final_safe_checkpoint.npz", allow_pickle=False) as checkpoint:
    errors = {}
    for name in sorted(expected_edits):
        assert name in checkpoint.files, (name, checkpoint.files)
        current = as_array(load(name))
        saved = checkpoint[name]
        assert current.shape == saved.shape
        errors[name] = float(np.max(np.abs(current - saved)))
        assert np.array_equal(current, saved), name
print("Checkpoint maximum errors:", errors)

def read_rollout(forward, model):
    path = Path(f"/work/fwd_{forward}/dose_edit_{model}.csv")
    df = pd.read_csv(path)
    columns = {str(c).strip().lower(): c for c in df.columns}

    def find_column(*aliases):
        for alias in aliases:
            if alias in columns:
                return columns[alias]
        raise AssertionError(f"Missing {aliases}; columns={list(df.columns)}")

    stay_col = find_column("stay_id", "stay", "stayid")
    edit_col = find_column("edit", "dose_edit", "intervention")
    shift_col = find_column("shift", "mean_shift", "predicted_shift", "delta")
    reached_col = find_column("reached", "drug_reached", "eligible")

    reached = (
        df[reached_col].astype(str).str.strip().str.lower()
        .isin(["true", "1", "1.0", "yes", "t"])
    )
    df = df.loc[reached]
    return pd.DataFrame({
        "stay": df[stay_col].to_numpy(),
        "edit": df[edit_col].to_numpy(),
        "shift": pd.to_numeric(df[shift_col], errors="raise").to_numpy(),
        "forward": forward,
        "model": model,
    })

rollouts = pd.concat(
    [read_rollout(k, model)
     for k in [43, 44, 49, 50]
     for model in ["original", "edited"]],
    ignore_index=True,
)
summaries = {}
expected_counts = {
    "insulin_x2": 2009, "insulin_half": 2009,
    "ne_x2": 1080, "ne_half": 1080,
}
for model in ["original", "edited"]:
    df = rollouts.loc[rollouts["model"].eq(model)]
    summary = df.groupby("edit").agg(
        hour_weighted_shift=("shift", "mean"),
        n_stay_hours=("shift", "size"),
        n_stays=("stay", "nunique"),
    )
    stay_means = df.groupby(["edit", "stay"])["shift"].mean()
    summary["mean_of_stay_means"] = stay_means.groupby(level="edit").mean()
    assert summary["n_stay_hours"].to_dict() == expected_counts
    summaries[model] = json.loads(summary.to_json(orient="index"))
    print(f"\n{model.upper()}: ALL 400 TUNING STAYS")
    print(summary.to_string())

factual = {
    "n_stays": 400,
    "n_readings": 18654,
    "original_nll": 1.6293686555162432,
    "edited_nll": 1.6326800847003322,
    "delta": 0.0033114291840892047,
    "stay_cluster_se": 0.0007908524903413655,
    "upper95": 0.004861500065158281,
}
report = {
    "decision": "Retain the exact validated checkpoint; no weights changed.",
    "forward_hash": "864804d79e",
    "checkpoint_errors": errors,
    "factual": factual,
    "dose_rollouts_all400": summaries,
    "caution": (
        "Rollout shifts are model predictions, not causal ground truth. "
        "Norepinephrine magnitude remains below provisional record-derived targets."
    ),
}
Path("/work/final_validation_after_f50.json").write_text(
    json.dumps(report, indent=2)
)
with Path("/work/notes.md").open("a") as notebook:
    notebook.write(
        "\n## Run 280 / Forward 50: final submission audit\n"
        "Retained the exact validated insulin .90 / NE .35 checkpoint with "
        "shrink80-scale4 factual cancellation. No weights changed. All five "
        "edited tensors match final_safe_checkpoint.npz exactly.\n"
        "Final cohort retained all four response directions. Pooled exact "
        "reached-hour rollout results over all400 tuning stays:\n"
        + json.dumps(summaries, indent=2)
        + "\nFactual validation remains:\n"
        + json.dumps(factual, indent=2)
        + "\nTreatment corrections were derived from training records, not "
        "medical knowledge. Predictions do not establish causal truth; "
        "NE magnitude remains uncertain. The stronger NE trial was rejected "
        "to preserve the precommitted factual-fit safety margin.\n"
    )
print("\nFinal report and notebook updated. No weights changed.")
```