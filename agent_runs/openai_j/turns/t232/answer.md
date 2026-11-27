The MAP repair uses two reserved value-head neurons to add a gated linear correction; the rest of the head, including the insulin repair, is retained. I’ll compare its local norepinephrine responses with the original head and recover the contrast settings. If the correction is already strong locally but weak in full rollouts, simply amplifying it may waste factual-fit margin.

```python
import torch
import numpy as np

print("=== TARGET SUMMARY ===")
print(Path("/work/kernel_targets.csv").read_text()[:1500])

print("\n=== CAUSAL ESTIMATION NOTES ===")
p206 = notes204.find("## Factual baseline and latent dynamics")
print(notes204[p206:p206+3400])

print("\n=== CONTRAST AND CALIBRATION VARIABLE MAP ===")
for prefix206 in ("map172", "map192", "cal190"):
    for n206, o206 in list(globals().items()):
        if n206.startswith(prefix206) and not inspect.isfunction(o206):
            if isinstance(o206, dict):
                print(n206, "keys", list(o206)[:18])
            else:
                print(n206, type(o206).__name__, getattr(o206, "shape", ""),
                      repr(o206)[:90] if not hasattr(o206, "shape") else "")

current206 = {n: load(n).clone() for n in head177_names}
original206 = {n: load_original(n).clone() for n in head177_names}
expected206 = head177_make(map192_candidates[(10, 6)])
print("\nCurrent versus saved candidate:",
      {n: float((current206[n]-expected206[n]).abs().max())
       for n in head177_names})

base206 = {}
for label206, w206 in (("original", original206), ("edited", current206)):
    base206[label206] = head177_eval(wm_train145, w206)["mean"]

print("\n=== LOCAL CONTRASTS, NO NEW EDIT ===")
events206 = head_meta["event"].to_numpy()
for cn206 in ("cf146", "cf172"):
    mc206 = globals()[cn206]
    dr206 = np.asarray(mc206["delta_rate"])
    act206 = np.asarray(mc206["active"], dtype=bool)
    print(cn206, "delta_rate shape", dr206.shape,
          "active count", int(act206.sum()),
          "delta-rate quantiles", np.nanquantile(dr206[act206], [0, .1, .5, .9, 1]))
    for label206, w206 in (("original", original206), ("edited", current206)):
        mu206 = head177_eval(mc206, w206)["mean"]
        for group206 in ("glucose", "map"):
            m206 = act206 & np.char.startswith(events206.astype(str), group206)
            if m206.any() and dr206.ndim == 1:
                shift206 = mu206[m206] - base206[label206][m206]
                slope206 = float(np.dot(shift206, dr206[m206]) /
                                 max(np.dot(dr206[m206], dr206[m206]), 1e-12))
                print(cn206, label206, group206, "n", int(m206.sum()),
                      "mean shift", round(float(shift206.mean()), 5),
                      "dose slope", round(slope206, 5))
```
