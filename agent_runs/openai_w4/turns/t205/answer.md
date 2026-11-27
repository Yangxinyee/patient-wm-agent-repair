The old norepinephrine decoder can respond to dose changes, but its gate often switches off under halved doses. I’ll refit that gate using factual and dose-perturbed encodings, with stays held out. Fewer missed drug bundles without more non-drug activation would support reusing the reserved head.

```python
from sklearn.svm import LinearSVC
from pathlib import Path
import numpy as np, inspect

ix180 = np.asarray(indices102, dtype=int)
active_ne180 = np.asarray(reference_bank125["has_norepinephrine"])[ix180].astype(bool)
sids180 = np.asarray(reference_bank125["stay_id"])[ix180]
train180 = sids180 < np.sort(np.unique(sids180))[75]
test180 = ~train180
raws180 = [np.asarray(raw87)] + [np.asarray(raw_cf87[k]) for k in
           [("insulin", 0.5), ("insulin", 2.0),
            ("norepinephrine", 0.5), ("norepinephrine", 2.0)]]
xx180 = np.concatenate([x[:, 384:] for x in raws180])
yy180 = np.tile(active_ne180.astype(int), len(raws180))
tr180 = np.tile(train180, len(raws180))
te180 = ~tr180
gate_fits180 = {}
for c180 in [0.1, 1.0, 10.0, 100.0]:
    g180 = LinearSVC(C=c180, class_weight="balanced", dual=False,
                    max_iter=20000, tol=1e-7)
    g180.fit(xx180[tr180], yy180[tr180])
    gate_fits180[c180] = g180
    pred180 = g180.decision_function(xx180[te180])
    y180 = yy180[te180].astype(bool)
    print("GATE", c180, "held accuracy", np.mean((pred180 > 0) == y180),
          "active missed", np.mean(pred180[y180] <= 0),
          "inactive fired", np.mean(pred180[~y180] > 0),
          "active q05", np.quantile(pred180[y180], .05),
          "inactive q99", np.quantile(pred180[~y180], .99))

print("\nNE DECODER TYPES")
for key180 in [("norepinephrine", .001, 10.0),
               ("norepinephrine", .01, 10.0), ("norepinephrine", .1, 10.0)]:
    v180 = raw_candidates87[key180]
    print(key180, type(v180).__name__, getattr(v180, "shape", None),
          list(v180) if isinstance(v180, dict) else "",
          str(v180)[:130])

print("\nLATENT NORMALIZATION")
print(inspect.signature(norm_latent104),
      norm_latent104.__code__.co_names, norm_latent104.__code__.co_consts)
print("norm tensors", [n for n in NAMES
                       if n.startswith("core.prelude.0") and "norm" in n])
print("z_in bias norm", np.linalg.norm(arr177(load("z_in.bias"))),
      "carrier loading", carrier104 @ arr177(load("z_in.weight")) @ latent_axis104)
print("least original hidden importance",
      [(int(j), float(importance94[j])) for j in
       np.argsort(importance94)[:12]])

print("\nRESPONSE DESIGN")
for k180 in ["JJ118", "J110", "target118", "probe_events118",
             "base_state118", "directions118", "probe_data95"]:
    if k180 in globals():
        v180 = globals()[k180]
        print(k180, getattr(v180, "shape", None),
              list(v180)[:8] if isinstance(v180, dict) else "")
print("dose-change encoding functions",
      [(k, str(inspect.signature(v))) for k, v in list(globals().items())
       if callable(v) and hasattr(v, "__code__")
       and "atom_exact65" in v.__code__.co_names])

with Path("/work/notes.md").open("a") as f180:
    f180.write("\n\n## Run 180\n"
               "Refitted NE action gates on factual and perturbed encodings, evaluating on "
               "held-out stays. Existing gate misses under half doses motivate this change. "
               "Recovered latent-normalization and response-design interfaces. No weights saved.\n")
```