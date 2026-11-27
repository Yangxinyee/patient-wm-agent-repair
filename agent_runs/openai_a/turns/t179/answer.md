The existing graft scales a residual-dose signal by the head’s local mean sensitivity. Its weak rollout response may therefore reflect poor dose decoding, rather than simply too small an output coefficient. I’ll check that decoding directly and recompute the insulin target with stay-level weighting. Strong, monotone synthetic-dose responses would support a revised graft; weak or inconsistent responses would refute it.

```python
def matched_causal_targets163(cube, beta, rho, drug_col, outcome_col):
    n = len(cube)
    shift = np.zeros(n, dtype=np.float64)
    reached = np.zeros(n, dtype=bool)
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    for hour in range(13, 25):
        dose_h = np.nan_to_num(cube[:, hour, drug_col], nan=0.0)
        shift = rho * shift + beta * dose_h
        reached |= dose_h > 0
        eligible = reached & np.isfinite(cube[:, hour + 1, outcome_col])
        sums[eligible] += shift[eligible]
        counts[eligible] += 1
    per_stay = sums[counts > 0] / counts[counts > 0]
    return {"mean_x2": float(per_stay.mean()), "mean_half": float(-.5 * per_stay.mean()),
            "eligible_stays": len(per_stay), "eligible_hours": int(counts.sum()),
            "stay_sd": float(per_stay.std())}

print("STAY-WEIGHTED INSULIN TARGETS")
for label, cube in [("train", record_cube160),
                    ("tuning_all", tpanel.to_numpy().reshape(400, 26, 6)),
                    ("tuning_first100", tpanel.to_numpy().reshape(400, 26, 6)[:100])]:
    print(label, matched_causal_targets163(cube, *rd_dynamic_fit["x"], 2, 0))

print("STATE POOL", rm_states["x"].shape,
      len(np.unique(rm_states["stay_id"])),
      np.unique(rm_states["stay_id"])[:3])
print("BUNDLE POOL", len(rec_bundle_index),
      len(set(s for s, h in rec_bundle_index)))
print("MEMORY OPTION TYPE",
      type(r131_memory_options[(True, 3600.0, 1.0, "both_input")]).__name__)
print("MEMORY OPTION STRUCTURE",
      brief_structure(r131_memory_options[(True, 3600.0, 1.0, "both_input")]))
print("=== MEMORY BUILDER ===")
print(compact_dis160(rec_memory)[:2400])
print("=== NOREPINEPHRINE REGRESSION CONTROLS ===")
print(compact_dis160(make_ne_data)[:3200])
print("=== INPUT CACHE FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if isinstance(v, types.FunctionType) and k.startswith(("sc_", "repair_")):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount],
              v.__code__.co_names)
```
