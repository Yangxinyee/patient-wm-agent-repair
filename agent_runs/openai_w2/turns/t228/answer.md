The discrepancy is much larger in non-action bundles than in drug bundles. That makes a general state-ordering error less likely, but does not establish the cause. I’ll check cached weights, time-flow calculations, and the worst measurement bundles. A changed cache or context mismatch would explain it; neither would mean the observation encoding needs closer examination.

```python
import torch

print("CACHED WEIGHTS VS ORIGINAL")
r196_changed_cache = []
for r196_name, r196_tensor in Wrepair.items():
    if r196_name in SHAPES:
        r196_np = (
            r196_tensor.detach().cpu().numpy()
            if isinstance(r196_tensor, torch.Tensor)
            else np.asarray(r196_tensor)
        )
        r196_original = load_original(r196_name)
        if not np.array_equal(r196_np, r196_original):
            r196_changed_cache.append(
                (r196_name, float(np.max(np.abs(r196_np - r196_original))))
            )
print(r196_changed_cache)

print("\nPOSTERIOR / GAP TENSOR NAMES")
print([(n, SHAPES[n]) for n in NAMES if any(
    s in n for s in ("z_post", "gap.", "flow.a", "out_norm")
)])

print("\nTIME FLOW CHECK")
r196_rows = np.asarray(rows188)
r196_dt = torch.tensor(
    (S187["hour"][r196_rows] - S187["hour"][r196_rows - 1]) * 3600,
    dtype=torch.float32
)
r196_prev_h = sx188[r196_rows - 1, 192:960].reshape(-1, 4, 192)
r196_hf = r196_prev_h * torch.exp(
    -r196_dt[:, None, None] * Wrepair["core.flow.a"].exp()[None, None, :]
)
print("recomputed h-flow vs hf188 RMS:",
      float(torch.mean((r196_hf - hf188) ** 2).sqrt()))
print("row selector valid:", bool(np.all(
    S187["stay_id"][r196_rows] == S187["stay_id"][r196_rows - 1]
)))

print("\nPOSTERIOR ERROR BY CURRENT BUNDLE")
r196_error = torch.mean((z_fit191 - z188) ** 2, dim=1).sqrt().numpy()
r196_summary = meta188.iloc[r196_rows].copy()
r196_summary["error"] = r196_error
r196_summary["dt_seconds"] = r196_dt.numpy()
r196_summary["action"] = (
    r196_summary.has_insulin | r196_summary.has_norepinephrine
)
print(r196_summary.groupby("action").error.agg(["size", "mean", "max"]))
print(r196_summary.groupby("n_atoms").error.agg(["size", "mean", "max"]))

print("\nWORST BUNDLES AND THEIR RECORDS")
for r196_row in r196_summary.nlargest(4, "error").itertuples():
    print("bundle", r196_row.stay_id, r196_row.hour, "error", r196_row.error)
    r196_record_subset = records188[
        (records188.stay_id == r196_row.stay_id)
        & (
            np.isclose(records188.hour, r196_row.hour)
            | np.isclose(records188.end_hour, r196_row.hour)
        )
    ]
    print(r196_record_subset.to_string(index=False))

print("\nNUMERIC SPEC CHECK")
print("numeric type:", type(VOCAB["numeric"]).__name__)
print(str(VOCAB["numeric"])[:1500])
print("local MAP specification:", numeric_specs.get(("map_arterial", "value")))
print("\nEDIT UNCHANGED:",
      np.array_equal(load("core.b_in.weight"), validated_B176))
```
