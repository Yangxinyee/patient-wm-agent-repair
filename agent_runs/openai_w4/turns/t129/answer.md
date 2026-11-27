The amplitude-80 channel has the lowest held-out leakage so far, about 0.18 in decoded innovation units. That is encouraging, but it is not a factual-fit result. I’ll inspect the measured state-to-reading sensitivities and causal lag estimates before connecting the channel to an outcome. The connection needs to affect glucose selectively and remain stable.

```python
import numpy as np
print("=== TRANSITION-SENSITIVITY OBJECTS ===")
for nm, ob in sorted(list(globals().items())):
    if any(s in nm for s in ["109", "110"]) and not nm.startswith("_"):
        if hasattr(ob, "shape"):
            ar0 = np.asarray(ob)
            if ar0.dtype.kind in "fiu":
                print(nm, ar0.shape, "rms", round(float(np.sqrt(np.mean(ar0.astype(float)**2))), 6),
                      "range", (round(float(ar0.min()), 6), round(float(ar0.max()), 6)))
        elif isinstance(ob, (str, int, float, list, tuple, dict)):
            print(nm, str(ob)[:160])
print("=== PROBE EVENTS ===")
print(probe_data95.groupby("event").size().to_dict())
print("probe indices", probe_indices95.shape, probe_indices95[:8])
print("J110 by event")
for ev0, ids0 in probe_data95.groupby("event").groups.items():
    j0 = J110[np.asarray(list(ids0), dtype=int)]
    print(ev0, j0.shape, "mean norm", float(np.linalg.norm(j0.mean(0))),
          "rms", float(np.sqrt(np.mean(j0**2))))

print("=== AMPLITUDE-80 WEIGHT CHANGES ===")
chosen_channel116 = channel_fits112[80.0]
for nm, ww in chosen_channel116["weights"].items():
    o0 = np.asarray(load_original(nm))
    w0 = np.asarray(ww)
    if np.max(np.abs(w0 - o0)) > 1e-7:
        print(nm, w0.shape, "delta rms", float(np.sqrt(np.mean((w0-o0)**2))))
print("decoder norm", np.linalg.norm(chosen_channel116["decoder"]))
print("=== CAUSAL KERNEL OBJECTS ===")
for nm in ["impulse_tables_repair", "kernel_models88", "gate_models88"]:
    print(nm)
    for kk, ob in globals()[nm].items():
        if hasattr(ob, "to_string"):
            print(kk, ob.to_string(index=False)[:1500])
        elif isinstance(ob, dict):
            print(kk, {k: (np.asarray(v).tolist() if hasattr(v, "shape") and np.asarray(v).size < 30
                          else str(v)[:180]) for k, v in ob.items()})
        else:
            print(kk, str(ob)[:1200])
print("=== PANEL / SPLIT / RAW INPUT NAMES ===")
print(" | ".join(
    nm + (":" + str(tuple(ob.shape)) if hasattr(ob, "shape") else "")
    for nm, ob in sorted(list(globals().items()))
    if any(s in nm.lower() for s in ["panel", "raw", "fit94", "test94", "train94", "hold94", "innovation"])
    and not nm.startswith("_") and not callable(ob)
)[:2000])
```
