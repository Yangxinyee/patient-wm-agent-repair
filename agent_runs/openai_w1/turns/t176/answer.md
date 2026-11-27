The probe’s factual NLL increased by 0.00208 nats, within the allowed limit on this batch. It remains a fallback, not a complete repair. I’ll estimate accumulated effects for these rollout stays and prepare matched state exports; that will help distinguish insufficient drug encoding from insufficient response in the value head.

```python
print("=== ORIGINAL STATE EXPORT SCHEMA ===")
state4_path = "/work/fwd_4/states_original.npz"
state4_recovery = np.load(state4_path)
for key in state4_recovery.files:
    arr = state4_recovery[key]
    print(key, arr.shape, arr.dtype)
probe_train_ids = sorted(
    np.unique(state4_recovery["stay_id"]).astype(int).tolist()
)
print("=== NEXT STATE REQUEST ===")
print(json.dumps({
    "type": "states", "model": "edited", "split": "train",
    "stays": probe_train_ids, "bundles": "all"
}))

print("\n=== APPROXIMATE SIX-LAG EFFECT TARGETS ===")
tuning_recovery = pd.read_csv("/data/tuning.csv")
kernel_recovery = {
    "insulin": np.array([-1.85034014, -2.25611825, -1.94263882,
                        -1.56111897, -1.02224091, -0.69018551]),
    "norepinephrine": np.array([3.35057748, 2.62101256, 1.71717035,
                              0.94251408, 0.49378644, 0.40891321])
}
rollout_recovery = pd.read_csv("/work/fwd_9/dose_edit_original.csv")
for drug, edits_for_drug in (
    ("insulin", (("insulin_x2", 2.0), ("insulin_half", 0.5))),
    ("norepinephrine", (("ne_x2", 2.0), ("ne_half", 0.5)))
):
    drug_rows = tuning_recovery[tuning_recovery.event.eq(drug)]
    rate_by_hour = drug_rows.assign(
        exposure_hour=np.floor(drug_rows.hour + 1e-6).astype(int)
    ).groupby(["stay_id", "exposure_hour"]).rate.sum().to_dict()
    for edit_name, multiplier in edits_for_drug:
        rr = rollout_recovery[
            rollout_recovery.edit.eq(edit_name) &
            rollout_recovery.reached.astype(bool)
        ]
        shifts = []
        for row in rr.itertuples(index=False):
            h = int(row.hour)
            shift = sum(
                coefficient * rate_by_hour.get((row.stay_id, h - 1 - lag), 0.0)
                for lag, coefficient in enumerate(kernel_recovery[drug])
                if h - 1 - lag >= 12
            ) * (multiplier - 1.0)
            shifts.append(shift)
        print(edit_name, "six-lag linear estimate", float(np.mean(shifts)),
              "hours", len(shifts))

print("\n=== RESIDUAL CACHE SCHEMA ===")
for key, value in causal_residuals.items():
    if isinstance(value, dict):
        print(key, {k: getattr(v, "shape", type(v).__name__)
                    for k, v in value.items()})
    elif isinstance(value, (tuple, list)):
        print(key, [(type(v).__name__, getattr(v, "shape", None))
                    for v in value[:10]])
    else:
        print(key, type(value).__name__, getattr(value, "shape", None))

numeric_probe5_saved = torch.as_tensor(load("embed.numeric.weight")).clone()
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 9–10 / Run 160\n"
            "5x numeric dose/rate scaling: insulin x2 -2.449, half +2.651; "
            "NE x2 +0.043, half -0.323. Tuning factual NLL 1.53347 vs "
            "original 1.53139, delta +0.00208 (within bound on this batch). "
            "Keeping this as a reversible fallback while obtaining matched "
            "states and six-lag record-derived targets. "
            "Counts: 160 Python runs, 10 forward passes.\n")
```