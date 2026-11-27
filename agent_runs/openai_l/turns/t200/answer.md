The records support similar lagged effects across the two glucose sensors and across the two MAP sensors. The current route has a large weight change but little dose sensitivity at the value head. I’ll check its scaling and recover the baseline rollout results before choosing between a stronger wire and a value-head correction. Either must pass the factual audit.

```python
import dis, inspect, numpy as np, re

print("KEY NOTE ENTRIES")
for s in sections182:
    h = s.splitlines()[0]
    if h in ["Baseline rollout", "Strengthened record evidence (run 4)", "Run 17", "Run 18", "Run 19", "Run 117 / direct-attention controllability", "Run 118 / private-state readout feasibility", "Run 125 / private controller and attention wire"] or "Run 167" in h:
        print("\n##", s[:1500])

print("\nFUNCTION INVENTORY")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__" and hasattr(v, "__code__"):
        print(k, inspect.signature(v))

print("\nARRAY/DATA INVENTORY")
for k in ["input_edited173", "ctx_edited173", "hidden_edited173", "logits_edited173",
          "I180", "R180", "Vsignal118", "Jselected118", "SP125", "wire_e152",
          "wire_coords152", "chosen125", "events179", "stays179", "B179",
          "targets179", "original_nll179", "sid63", "hour63", "sx63",
          "train_df", "train", "df", "records", "r30_meta", "meta80"]:
    if k in globals():
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, type(v).__name__, v.shape,
                  "preview", str(v[:3])[:180])
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:15])
        else:
            print(k, str(v)[:220])

print("\nWRITE SCALING")
print("chosen125", chosen125, "amp167", amp167)
for name in ["core.delta_in", "core.b_in.weight", "core.c_out.weight"]:
    a = np.asarray(load_original(name))
    print(name, a.shape, "norm", float(np.linalg.norm(a)),
          "range", float(a.min()), float(a.max()))
print("R180 vs Vsignal118 principal overlaps",
      np.linalg.svd(np.linalg.qr(R180)[0].T @ np.linalg.qr(Vsignal118)[0], compute_uv=False))

print("\nHEAD INPUT DISASSEMBLY")
dis.dis(head_inputs96)
print("\nCURRENT WRITE NOTE\n",
      next((s for s in sections182 if "Run 167" in s.splitlines()[0]), "not found"))
```
