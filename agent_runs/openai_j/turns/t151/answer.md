The helper uses an eight-field register, while the specification describes it only schematically. I’ll test a matched pre-ICU reading and inspect the retained candidate weights. A mismatch there would rule out tuning-stay differences as the main explanation.

```python
print("=== NORMALISATION ENTRIES ===")
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    print(event, repair_norms[(event, "value")])

print("\n=== RETAINED RECENT CANDIDATE OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith(("r129_", "r130_", "r131_")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            a = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
            print(name, a.shape,
                  "range", (float(np.min(a)), float(np.max(a))) if a.size else None)
        elif np.isscalar(obj):
            print(name, obj)
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:12])

print("\n=== FIRST-STAY TIMING ===")
sel = tune_states["stay_id"] == 2020000000
print("state hours:", tune_states["hour"][sel][:12])
print(tuning[tuning.stay_id == 2020000000].head(10).to_string(index=False))

print("\n=== MATCHED INITIAL STATE ===")
train_initial_i = int(head_meta.iloc[0].state_i)
tune_initial_i = np.flatnonzero(sel)[0]
print("train and tuning initial hours:",
      repair_states["hour"][train_initial_i],
      tune_states["hour"][tune_initial_i])
print("state max difference:",
      np.max(np.abs(repair_states["x"][train_initial_i]
                    - tune_states["x"][tune_initial_i])))

audit_initial_meta = head_meta[
    (head_meta.hour == -1) & (head_meta.event == "glucose_finger")
].iloc[:1].copy()
audit_initial_meta["stay_id"] = 2020000000
audit_initial_meta["state_i"] = tune_initial_i
audit_initial_base = reconstruct_reading_head(
    audit_initial_meta, state_array=tune_states["x"]
)
audit_initial_base.update(corrected_register_head(
    audit_initial_meta, audit_initial_base
))
audit_centres = torch.as_tensor(
    repair_centres_for_meta(audit_initial_meta), dtype=torch.float32
)
print("CPU original initial mean:",
      (audit_initial_base["probs"] * audit_centres).sum(-1).numpy())
print("GPU initial glucose rows:")
print(tune_factual_original[
    (tune_factual_original.stay_id == 2020000000)
    & (tune_factual_original.hour == -1)
].to_string(index=False))
print("initial numeric fields:", audit_initial_base["numbers"].numpy())
```
