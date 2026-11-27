Some low-valued readings have very large errors, so ordinary sampling variation is unlikely to explain the mismatch. I’ll check the scratch weights, helper hooks, and exact registers at those rows. A leftover experimental weight or a register from the wrong bundle would explain why the purported local baseline differs from the original model.

```python
print("=== SCRATCH WEIGHT AUDIT ===")
changed103 = []
for n103 in NAMES:
    if n103 in W0:
        a103 = W0[n103]
        a103 = a103.detach().cpu().numpy() if isinstance(a103, torch.Tensor) else np.asarray(a103)
        z103 = load_original(n103)
        z103 = z103.detach().cpu().numpy() if isinstance(z103, torch.Tensor) else np.asarray(z103)
        d103 = float(np.max(np.abs(a103 - z103)))
        if d103 > 1e-7:
            changed103.append((n103, d103, float(np.linalg.norm(a103 - z103))))
print("non-original W0 tensors:", changed103)
print("helpers:", getattr(lin0, "__name__", None), getattr(mlp0, "__name__", None),
      getattr(norm0, "__name__", None))
print("=== WORST-ROW EXACT STATE/REGISTER ALIGNMENT ===")
for row103 in show102.nlargest(5, "abs_error").itertuples():
    pos103 = int(row103.position)
    ix103 = int(idx80[pos103])
    sj103 = r49_sensors.index(row103.event)
    past103 = df[(df.stay_id == row103.stay_id) &
                 (df.event == row103.event) & (df.hour < row103.hour)]
    print("target", row103.stay_id, row103.hour, row103.event,
          "position", pos103, "state index", ix103,
          "state key", (int(r49_stays[ix103]), float(r49_hours[ix103])),
          "register", r49_register[ix103, sj103].tolist(),
          "raw last", past103[["hour", "value"]].tail(1).values.tolist())
print("=== PREDICTIONS DIRECTLY FROM EXTERNAL STATES ===")
worst_pos103 = show102.nlargest(5, "abs_error").position.to_numpy(dtype=int)
worst_ix103 = idx80[worst_pos103]
xx103 = torch.tensor(sfile101["x"][worst_ix103], dtype=torch.float32)
saved_means103 = r49_readouts(xx103[:, :192], xx103[:, 192:960].reshape(-1, 4, 192),
                            worst_ix103)
print(saved_means103)
print("=== CORE CONTRACT ===")
print(inspect.signature(r59_core), r59_core.__code__.co_varnames)
print(r59_core.__code__.co_names)
print("=== NUMERIC SPEC EXAMPLES ===")
for sensor103 in ("glucose_finger", "map_arterial"):
    print(sensor103, num_spec0[(sensor103, "value")])
with open("/work/notes.md", "a") as f103:
    f103.write("\n## Run 103: original-weight and worst-row audit\n"
               "- Extreme errors on low readings are inconsistent with ordinary prior-sampling differences.\n"
               "- Checking scratch weights, active hooks, exact registers, and predictions from external states.\n"
               "- Python runs: 103; forward passes: 4; no submitted edits.\n")
```
