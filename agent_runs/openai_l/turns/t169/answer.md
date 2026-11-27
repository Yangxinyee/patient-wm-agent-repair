The recovered kernels imply an insulin response that builds over several hours and a faster norepinephrine response; the original rollouts are substantially smaller than the provisional record-based targets. I’ll now check the wire’s absolute factual signal and the transition implementation. A useful wire must preserve dose contrasts without injecting a large unrelated signal on recorded stays.

```python
import dis

def show_function157(func, cap=5000):
    try:
        text = inspect.getsource(func)
    except Exception:
        text = dis.Bytecode(func).dis()
    print(text[:cap])
    if len(text) > cap:
        print("[truncated; total characters:", len(text), "]")

print("=== WEIGHT ACCESS ===")
show_function157(w63, 1800)
print("=== CORE TRANSITION ===")
show_function157(transition64, 6200)
print("=== PRELUDE FEATURES ===")
show_function157(prelude_features102, 2600)

print("\n=== WIRE AMPLITUDES ===")
for name in ["base_e152", "wire_e152", "de152", "wire_coords152",
             "Ebase153", "newd125", "oldd125", "pooled125", "resid125"]:
    a = np.asarray(globals()[name])
    axes = tuple(range(a.ndim - 1))
    rms = np.sqrt(np.mean(a*a, axis=axes))
    print(name, "shape", a.shape, "RMS range",
          np.round([np.min(rms), np.median(rms), np.max(rms)], 5))
    if a.shape[-1] == 4:
        print("  channel RMS", rms)
for name in ["base_e152", "wire_e152"]:
    a = np.asarray(globals()[name])
    coords = a @ QE125
    print(name, "QE mean", coords.mean(axis=(0, 1)),
          "QE SD", coords.std(axis=(0, 1)))
print("wire projection difference RMS",
      np.sqrt(np.mean(((wire_e152-base_e152) @ QE125)**2, axis=(0, 1))))
print("chosen attention channel SD", newd125[:, chosen125].std(axis=0))
print("wire_out difference norm", np.linalg.norm(wire_out152-Wrepair125))

print("\n=== CAUSAL KERNELS ===")
for key, value in kernels22.items():
    print(key, np.asarray(value))
print("record targets", record_row_targets25)

print("\n=== SIMULATION DATA SHAPES ===")
for name in ["cache", "W", "weights", "bundles", "rows", "states",
             "ids152", "wire_ids153"]:
    if name in globals():
        print(name, brief156(globals()[name]))
```
