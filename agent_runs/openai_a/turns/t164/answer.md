Anchored and unanchored predictions agree closely, so the small reconstruction error is not driving these local contrasts. More importantly, both the original head and the graft lower glucose when insulin is halved—the wrong sign relative to the record-derived estimate. I’ll expand the contrast dataset across the available training stays and use it to fit a new decoder rather than strengthen the old one.

```python
lab_meta = head_meta_train[
    head_meta_train["event"].eq("glucose_finger") &
    head_meta_train["has_last"].astype(bool)
].copy()
lab_rows = lab_meta["state_row"].to_numpy(dtype=int)
lab_keep = (
    np.asarray(rm_states["has_insulin"])[lab_rows].astype(bool) &
    ~np.asarray(rm_states["has_norepinephrine"])[lab_rows].astype(bool) &
    (np.asarray(rm_states["n_atoms"])[lab_rows] == 1) &
    (lab_rows > 0)
)
lab_meta = lab_meta.loc[lab_keep].reset_index(drop=True)
lab_indices = lab_meta["state_row"].to_numpy(dtype=int)

lab_raw = rm_records[rm_records["event"].eq("insulin")].drop_duplicates(
    ["stay_id", "hour"]
).set_index(["stay_id", "hour"])
lab_keys = pd.MultiIndex.from_arrays([
    np.asarray(rm_states["stay_id"])[lab_indices],
    np.asarray(rm_states["hour"])[lab_indices]
])
lab_rates = lab_raw["rate"].reindex(lab_keys).to_numpy(dtype=np.float32)
lab_doses = lab_raw["dose"].reindex(lab_keys).to_numpy(dtype=np.float32)
lab_valid = np.isfinite(lab_rates) & np.isfinite(lab_doses)
lab_meta = lab_meta.loc[lab_valid].reset_index(drop=True)
lab_indices, lab_rates, lab_doses = (
    a[lab_valid] for a in (lab_indices, lab_rates, lab_doses)
)
assert len(lab_indices) > 50
assert np.all(
    np.asarray(rm_states["stay_id"])[lab_indices]
    == np.asarray(rm_states["stay_id"])[lab_indices - 1]
)

lab_prev = np.asarray(rm_states["x"])[lab_indices - 1]
lab_dt = (
    np.asarray(rm_states["hour"])[lab_indices]
    - np.asarray(rm_states["hour"])[lab_indices - 1]
) * 3600.0
lab_world = rm_empty_e[:1].expand(len(lab_indices), -1)
lab_memory, lab_age, lab_mask = rec_memory(lab_indices)
lab_ctx, _ = rec_context(lab_prev, lab_dt)
lab_simulated = {}

saved_lab_rates, saved_lab_doses = rm_rates.copy(), rm_doses.copy()
try:
    for multiplier in (1.0, 0.5, 2.0):
        rm_rates, rm_doses = lab_rates * multiplier, lab_doses * multiplier
        action0 = rm_set(
            rm_insulin_atoms(3, 2, "left", True, "gelu"),
            "encode_action", activation="gelu"
        )
        posterior0 = rm_mlp(
            torch.cat([lab_ctx, lab_world, action0], dim=1), "z_post"
        )[:, :32]
        if multiplier == 1.0:
            posterior_error0 = probe_rms(
                posterior0.numpy() - np.asarray(rm_states["x"])[lab_indices, 960:]
            )
            print("unchanged posterior RMSE:", posterior_error0)
            assert posterior_error0 < 1e-4
        result0 = r131_step(
            lab_prev, lab_dt, lab_world, action0, posterior0,
            lab_memory, lab_age, lab_mask, age_mode="both_input"
        )
        slots0 = next(v for v in result0 if v.ndim == 3)
        out0 = next(v for v in result0 if v.ndim == 2 and v.shape[1] == 192)
        lab_simulated[multiplier] = torch.cat(
            [out0, slots0.reshape(len(out0), -1), posterior0], dim=1
        ).numpy()
finally:
    rm_rates, rm_doses = saved_lab_rates, saved_lab_doses

lab_head_results = {}
for multiplier in (1.0, 0.5, 2.0):
    states0 = dict(rm_states)
    states0["x"] = np.array(rm_states["x"], copy=True)
    states0["x"][lab_indices] += lab_simulated[multiplier] - lab_simulated[1.0]
    lab_head_results[multiplier] = probe_head_eval(lab_meta, states0)

print("synthetic cases:", len(lab_meta),
      "training stays:", lab_meta["stay_id"].nunique())
print("recorded rate mean/SD:", float(lab_rates.mean()), float(lab_rates.std()))
for multiplier in (0.5, 2.0):
    delta0 = lab_head_results[multiplier]["mean"] - lab_head_results[1.0]["mean"]
    print("multiplier", multiplier,
          "original local mean shift:", float(delta0.mean()),
          "feature contrast RMS:", probe_rms(
              lab_head_results[multiplier]["features"]
              - lab_head_results[1.0]["features"]
          ))

print("factual feature shapes:",
      {k: tuple(v.shape) for k, v in robust_X.items()})
print("projector diagnostics:", {
    k: (tuple(v.shape), float(np.trace(np.asarray(v))))
    for k, v in robust_projectors.items()
})
```
