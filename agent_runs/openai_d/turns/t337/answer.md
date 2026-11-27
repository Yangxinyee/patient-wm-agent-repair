The fourth cohort’s increase is about 0.00159 nats, bringing the trial’s pooled average below the limit. Its uncertainty bound still determines acceptance. I’ll calculate that bound from paired readings, clustered by stay, and automatically restore the validated checkpoint if it exceeds +0.005.

```python
import json

def _r278_factual_audit(forwards):
    frames = []
    for k in forwards:
        folder = Path(f"/work/fwd_{k}")
        original = pd.read_csv(folder / "factual_original.csv")
        changed = pd.read_csv(folder / "factual_edited.csv")
        assert len(original) == len(changed)

        nll_columns = [c for c in original.columns if "nll" in c.lower()]
        nll_col = "nll" if "nll" in nll_columns else (
            "value_nll" if "value_nll" in nll_columns else
            nll_columns[0] if len(nll_columns) == 1 else None
        )
        assert nll_col is not None, original.columns.tolist()
        stay_col = next(c for c in ("stay_id", "stay") if c in original.columns)
        for c in (stay_col, "hour", "event", "code", "value", "true_value",
                  "value_bin", "true_bin", "bundle_index", "atom_index"):
            if c in original.columns and c in changed.columns:
                pd.testing.assert_series_equal(
                    original[c], changed[c], check_names=False
                )

        frame = pd.DataFrame({
            "stay_id": original[stay_col],
            "original_nll": original[nll_col].astype(float),
            "edited_nll": changed[nll_col].astype(float),
            "forward": k
        })
        frame["delta"] = frame["edited_nll"] - frame["original_nll"]
        if "event" in original.columns:
            frame["event"] = original["event"]
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True)
    assert pooled["stay_id"].nunique() == 400
    delta = pooled["delta"].mean()
    clusters = pooled.groupby("stay_id")["delta"].agg(["sum", "count"])
    residual = clusters["sum"] - delta * clusters["count"]
    n_clusters = len(clusters)
    se = np.sqrt(n_clusters / (n_clusters - 1) *
                 np.square(residual).sum()) / len(pooled)
    stats = {
        "n_stays": n_clusters,
        "n_readings": len(pooled),
        "original_nll": pooled["original_nll"].mean(),
        "edited_nll": pooled["edited_nll"].mean(),
        "delta": delta,
        "stay_cluster_se": se,
        "upper95": delta + 1.96 * se
    }
    return pooled, stats

_r278_safe_rows, _r278_safe_stats = _r278_factual_audit([39, 40, 41, 42])
_r278_trial_rows, _r278_trial_stats = _r278_factual_audit([45, 46, 47, 48])
print("SAFE CHECKPOINT:", _r278_safe_stats)
print("NE .40 TRIAL:", _r278_trial_stats)
print("\nTRIAL BY COHORT:")
print(_r278_trial_rows.groupby("forward")["delta"].agg(["count", "mean"]))

np.savez("/work/ne40_trial.npz",
         **{name: np.asarray(load(name), dtype=np.float32)
            for name in edited()})

with np.load("/work/safe_before_ne40.npz") as saved:
    _r278_safe_weights = {name: saved[name].copy() for name in saved.files}
with np.load("/work/candidate90_35_bin80x4.npz") as saved:
    assert set(saved.files) == set(_r278_safe_weights)
    for name in saved.files:
        np.testing.assert_array_equal(saved[name], _r278_safe_weights[name])

_r278_restore = _r278_trial_stats["upper95"] >= 0.005
if _r278_restore:
    assert _r278_safe_stats["upper95"] < 0.005
    for name in list(edited()):
        if name not in _r278_safe_weights:
            drop_edit(name)
    for name, tensor in _r278_safe_weights.items():
        save_edit(name, tensor)
    for name, tensor in _r278_safe_weights.items():
        np.testing.assert_array_equal(np.asarray(load(name)), tensor)
    _r278_decision = (
        "REJECTED stronger NE trial; restored exact validated insulin .90 / "
        "NE .35 checkpoint. Remaining rollouts will cover tuning cohorts 2 and 3."
    )
else:
    _r278_decision = (
        "Trial passes factual bound; not yet accepted. Remaining rollouts must "
        "improve NE relative to forwards 43 and 44 without materially harming insulin."
    )
print("\nDECISION:", _r278_decision)
print("Current edited tensors:", edited())
print("\nDOSE FILE SCHEMA:")
print(pd.read_csv("/work/fwd_43/dose_edit_edited.csv").head(3).to_string(index=False))

with Path("/work/notes.md").open("a") as notebook:
    notebook.write(
        "\n## Run 278 / Forwards 45–48: stronger-NE factual decision\n"
        "Paired reading-weighted deltas; uncertainty clustered by stay using "
        "delta-sum minus mean-delta times reading count, with finite-cluster correction.\n"
        f"Safe: {json.dumps(_r278_safe_stats)}\n"
        f"NE .40 trial: {json.dumps(_r278_trial_stats)}\n"
        f"{_r278_decision}\n"
        "Trial backup: /work/ne40_trial.npz. Safe restoration, if triggered, "
        "was verified tensor-for-tensor against both saved safe files.\n"
    )
```