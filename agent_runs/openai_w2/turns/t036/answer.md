The probe comparison is not valid yet: its printed configuration differs from the one that passed the earlier reconstruction audit. I’ll freeze the documented register columns and GELU activations, then recompute both sides with identical draws. Near-zero channel biases must return before these features can support an edit.

```python
repair_head_cfg = {
    "activation": "gelu",
    "gap_mode": "seconds_div10",
    "type_id": 1,
    "status_id": 0,
    "age_mode": "seconds_div10",
    "zcol": 6,
    "acol": 3,
    "flags": (0,),
    "gap_activation": "gelu",
    "prior_activation": "gelu",
    "value_activation": "gelu",
}
saved_head_state_global = local_value_head.__globals__["S"]
try:
    local_value_head.__globals__["S"] = S_reference
    repair_original_features = local_value_head(
        head_data, repair_head_cfg, bin_side="left", zmode="prior_mc",
        nsamples=64, return_features=True, seed=917
    )
    local_value_head.__globals__["S"] = S_probe
    repair_probe_features = local_value_head(
        head_data, repair_head_cfg, bin_side="left", zmode="prior_mc",
        nsamples=64, return_features=True, seed=917
    )
finally:
    local_value_head.__globals__["S"] = saved_head_state_global

repair_feature_audit = head_data[["event"]].copy()
repair_feature_audit["mean_bias"] = (
    np.asarray(repair_original_features["mean"])
    - head_data["predicted_mean"].to_numpy()
)
repair_feature_audit["nll_bias"] = (
    np.asarray(repair_original_features["nll"])
    - head_data["nll"].to_numpy()
)
repair_feature_audit["dose_probe_shift"] = (
    np.asarray(repair_probe_features["mean"])
    - np.asarray(repair_original_features["mean"])
)
repair_delta_hidden = (
    np.asarray(repair_probe_features["hidden"])
    - np.asarray(repair_original_features["hidden"])
)
repair_feature_audit["probe_hidden_norm"] = np.linalg.norm(repair_delta_hidden, axis=1)
print(repair_feature_audit.groupby("event").agg(["mean", "std"]))
print("Difference from saved original-feature means:",
      float(np.sqrt(np.mean(
          (np.asarray(repair_original_features["mean"])
           - np.asarray(head_original_features["mean"]))**2))))
print("Current edits:", edited())

print("\n=== PERSISTENCE FUNCTION STRUCTURE ===")
try:
    print(inspect.getsource(sensor_iv_persistence)[:3000])
except Exception:
    print("Names:", sensor_iv_persistence.__code__.co_names)
    print("Constants:", sensor_iv_persistence.__code__.co_consts[:35])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 31: configuration aliasing fixed\n"
            "Run 30 used a mutable head_cfg that did NOT match the validated reconstruction; "
            "its reported probe differences must not be used. Frozen repair_head_cfg explicitly: "
            "all GELU, type 1, status 0, flags (0,), age col 3, value-z col 6, "
            "log1p(seconds)/10, bin side left. Recomputed repair_original_features and "
            "repair_probe_features with matched 64 prior draws (seed 917). "
            "Use repair_delta_hidden, not head_probe_delta_hidden, for fitting.\n")
```