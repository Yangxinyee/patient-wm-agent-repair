The helper lets us study dose changes relative to saved states without first reproducing every state exactly. I’ll test a more targeted hypothesis: the posterior contains dose information, but does not transmit the treatment innovation strongly enough. A decoder must recover dose minus its history-based prediction on held-out stays and remain quiet on non-drug bundles.

```python
print("--- Dose-encoder probe ---")
for drug, (grid, enc) in dose_encoder_probe.items():
    print(drug, "rates", grid,
          "distance from first", np.round(np.linalg.norm(enc - enc[0], axis=-1), 3))

print("\n--- Existing action and END diagnostics ---")
for label, obj in [
    ("action_probe_exact_doses", action_probe_exact_doses),
    ("end_convention_trials", end_convention_trials),
    ("measurement_unit_local", measurement_unit_local),
]:
    print(label)
    for k, v in obj.items():
        if isinstance(v, dict):
            print(" ", k, {kk: (tuple(vv.shape) if hasattr(vv, "shape") and not callable(vv.shape)
                               else repr(vv)[:100]) for kk, vv in v.items()})
        else:
            print(" ", k, repr(v)[:1200])

print("\n--- Data schemas ---")
print("panel", panel.index.names, panel.columns.tolist(), panel.head(3).to_string())
print("X", X.columns.tolist())
print("fA", fA.columns.tolist(), fA.head(3).to_string(index=False))
print("states4", {k: np.asarray(v).shape for k, v in s4.items()})
print("measurement_atom_local defaults", measurement_atom_local.__defaults__)
print("set_encode_single_local bytecode",
      "; ".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(set_encode_single_local)
                if i.opname not in {"RESUME", "PRECALL", "CACHE", "PUSH_NULL"}))
print("wm_set bytecode",
      "; ".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(wm_set)
                if i.opname not in {"RESUME", "PRECALL", "CACHE", "PUSH_NULL"}))
```
