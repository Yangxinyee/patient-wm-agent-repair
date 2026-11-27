The local transition matches the documented recurrence, but the verbose inspection output hid the wire diagnostics. I’ll keep this check compact and measure the wire’s baseline signal directly. That will tell us whether it is safe to connect to the private state channels or needs further factual cancellation.

```python
print("=== WIRE AMPLITUDES ===")
for name in ["base_e152", "wire_e152", "de152", "wire_coords152",
             "newd125", "oldd125", "resid125"]:
    a = np.asarray(globals()[name])
    axes = tuple(range(a.ndim - 1))
    rms = np.sqrt(np.mean(a*a, axis=axes))
    print(name, a.shape, "RMS min/median/max",
          np.round([rms.min(), np.median(rms), rms.max()], 6))
for name in ["base_e152", "wire_e152"]:
    coords = np.asarray(globals()[name]) @ QE125
    print(name, "QE mean", np.round(coords.mean(axis=(0, 1)), 6),
          "SD", np.round(coords.std(axis=(0, 1)), 6))
print("wire delta QE RMS",
      np.round(np.sqrt(np.mean(((wire_e152-base_e152) @ QE125)**2,
                              axis=(0, 1))), 6))
print("attention channel SD", newd125[:, chosen125].std(axis=0))
print("wire_out delta norm", np.linalg.norm(wire_out152-Wrepair125))

print("\n=== CAUSAL KERNELS / TARGETS ===")
for key, value in kernels22.items():
    print(key, np.round(np.asarray(value), 5))
print("targets", record_row_targets25)

print("\n=== COMPACT HELPER INTERFACES ===")
for func in [prepare103, function107, bundle_inputs74, attention_features116,
             head_inputs96, decode67, projected_flow153]:
    c = func.__code__
    print(func.__name__, "locals:", c.co_varnames)
    print(" globals:", c.co_names)
    print(" string constants:", [v for v in c.co_consts if isinstance(v, str)])

print("\n=== SUPPORTING ARRAYS ===")
g158 = dict(globals())
for name, obj in sorted(g158.items()):
    if name.endswith(("63", "64", "67", "96", "102", "103", "107")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:12])
        elif hasattr(obj, "columns"):
            print(name, obj.shape, list(obj.columns)[:15])

print("\n=== ORIGINAL PRIVATE OUTPUT ===")
C0_158 = np.asarray(load_original("core.c_out.weight"), dtype=np.float64)
p0_158 = C0_158.T @ R118
print("original private mean", hpmean153 @ p0_158)
print("original private centered SD", np.sqrt(np.mean((hpc153 @ p0_158)**2, axis=0)))
print("controller mean (.01)", hpmean153 @ controllers153[0.01])
```
