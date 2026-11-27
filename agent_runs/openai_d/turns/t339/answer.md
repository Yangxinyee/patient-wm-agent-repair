The restored checkpoint behaves consistently on cohort three: insulin shifts glucose by −13.95/+6.79, and norepinephrine shifts MAP by +1.35/−0.70. Insulin is close to the record-derived target; norepinephrine is directionally repaired but still too weak. I’ll record the pooled results and verify the final weights before checking the last cohort.

```python
_r279_rollouts = {}
_r279_tables = {}
for model in ("original", "edited"):
    parts = []
    for k in (43, 44, 49):
        frame = pd.read_csv(f"/work/fwd_{k}/dose_edit_{model}.csv")
        reached = frame["reached"].astype(str).str.lower().isin(["true", "1"])
        frame = frame.loc[reached & frame["hour"].between(13, 24)].copy()
        frame["forward"] = k
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    assert not rows.duplicated(["stay_id", "hour", "edit"]).any()
    _r279_rollouts[model] = rows

    table = rows.groupby("edit").agg(
        hour_weighted_shift=("shift", "mean"),
        n_stay_hours=("shift", "size"),
        n_stays=("stay_id", "nunique")
    )
    table["mean_of_stay_means"] = (
        rows.groupby(["edit", "stay_id"])["shift"].mean()
        .groupby("edit").mean()
    )
    _r279_tables[model] = table
    print(f"\n=== {model.upper()}: FIRST 300 TUNING STAYS ===")
    print(table.to_string())
    if model == "edited":
        print("\nEDITED BY COHORT:")
        print(rows.groupby(["forward", "edit"])["shift"]
              .agg(["count", "mean"]).to_string())

print("\n=== FINAL CHECKPOINT INTEGRITY ===")
assert set(edited()) == set(_r278_safe_weights)
for name, tensor in _r278_safe_weights.items():
    current = np.asarray(load(name), dtype=np.float32)
    np.testing.assert_array_equal(current, tensor)
    change = current.astype(np.float64) - np.asarray(
        load_original(name), dtype=np.float64
    )
    print(name, "original-change norm", float(np.linalg.norm(change)),
          "safe-checkpoint error", float(np.max(np.abs(current-tensor))))
np.savez("/work/final_safe_checkpoint.npz",
         **{name: np.asarray(load(name), dtype=np.float32)
            for name in edited()})

_r279_report = {
    "checkpoint": "864804d79e",
    "factual_all400": {k: float(v) for k, v in _r278_safe_stats.items()},
    "dose_first300": {
        model: {
            edit: {key: float(value) for key, value in row.items()}
            for edit, row in table.iterrows()
        }
        for model, table in _r279_tables.items()
    },
    "final_forward_plan": "dose_edit both, tuning IDs 2020000300..2020000399",
    "decision": "Submit validated insulin .90 / NE .35 with shrink80-scale4 cancellation.",
    "limitations": (
        "Cumulative causal targets are provisional record-derived estimates, "
        "not simulator truth. Norepinephrine magnitude remains under-repaired."
    )
}
Path("/work/final_validation_before_f50.json").write_text(
    json.dumps(_r279_report, indent=2)
)

with Path("/work/notes.md").open("a") as notebook:
    notebook.write(
        "\n## Run 279 / Forward 49: final checkpoint and rollout audit\n"
        "Forward49 confirmed restored hash 864804d79e. No weights changed.\n"
        "Final weights verified exactly against safe_before_ne40 and saved to "
        "/work/final_safe_checkpoint.npz. Edited reading-head layers and "
        "previous-value transition logits implement training-record-derived "
        "dose-innovation corrections with factual cancellation.\n"
        "Immediate effects were estimated with pre-treatment history adjustment "
        "and nonlinear cross-fitted nuisance models: insulin approximately "
        "-1.99 glucose units per rate unit; NE approximately +3.29 MAP units "
        "per rate unit. Controlled multi-hour kernels suggest larger cumulative "
        "effects, but those targets remain provisional.\n"
        f"All400 factual validation: {json.dumps(_r279_report['factual_all400'])}\n"
        "Reached-hour rollout aggregate over first300 tuning stays:\n"
        f"{json.dumps(_r279_report['dose_first300'])}\n"
        "NE .40 trial rejected because factual upper95 .005234 exceeded the "
        "precommitted .005 bound; safe checkpoint upper95 .004862 passes.\n"
        "Insulin rollout repair transfers across three cohorts. NE direction "
        "is repaired, but magnitude is still below provisional causal targets. "
        "No claim of complete causal recovery is warranted.\n"
        "Final remaining request: forward50, exact-checkpoint paired dose rollout "
        "on tuning IDs 2020000300..2020000399. No further edits planned. "
        "Final response will report expected shifts using all four cohorts, "
        "with factual increase +.003311 nats per reading.\n"
    )
print("\nFinal report saved; notebook updated. No weights changed.")
```