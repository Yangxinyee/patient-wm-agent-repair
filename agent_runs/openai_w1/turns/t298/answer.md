The architecture confirms that `core.b_in` can inject the dose signal directly into the recurrent state, bypassing the attention route that failed. The factual and dose-edited features are already cached. I’ll check their scaling and the local readout directions before fitting an edit; correct signs alone would not establish factual safety or rollout accuracy.

```python
import inspect
import numpy as np
import torch
from pathlib import Path

def arr262(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== INJECTION DIRECTIONS ===")
print("b_in shapes:", SHAPES.get("core.b_in.weight"), SHAPES.get("core.b_in.bias"))
print("base257:", base257)
print("T257:", T257)
print("B257 RMS:", np.sqrt(np.mean(B257**2, axis=0)))
for j in range(2):
    for scale in (-1.0, 1.0):
        print("local response", j, scale,
              arr262(direct_readout257(scale * B257[:, j])) - arr262(base257))

print("\n=== FUNCTIONS: NO DEFAULT-ARGUMENT PRINTING ===")
for name in ("E258", "direct_readout257", "_rxlinear257", "posterior238"):
    fn = globals().get(name)
    if not callable(fn):
        continue
    try:
        text = inspect.getsource(fn)
    except (OSError, TypeError):
        c = fn.__code__
        text = ("arguments=" + repr(c.co_varnames[:c.co_argcount]) +
                "\nnames=" + repr(c.co_names) +
                "\nconstants=" + repr(tuple(x for x in c.co_consts
                                          if isinstance(x, (str, int, float))))[:1000])
    print(name, text[:2300])

print("\n=== ALIGNED FEATURE AND DOSE CHECKS ===")
print("factual E:", factualE258.shape,
      "RMS", np.sqrt(np.mean(factualE258**2)))
for j, drug in enumerate(("insulin", "norepinephrine")):
    mask = np.asarray(ds_s["has_" + drug], dtype=bool)
    rates = ds_rates[mask, j]
    print(drug, "bundles", mask.sum(), "rate quantiles",
          np.quantile(rates, [0, .1, .5, .9, 1]))
    for mult in (0.0, 0.5, 2.0, 3.0):
        ec = arr262(Es259[(drug, mult)])
        de = ec - factualE258
        print(" multiplier", mult, "shape", ec.shape,
              "contrast RMS", np.sqrt(np.mean(de[mask]**2)),
              "mean physical rate change", np.mean(rates*(mult-1)))
print("ds_eval_ids:", list(ds_eval_ids)[:15])
print("ds_s keys:", list(ds_s))
print("safe236 keys:", list(safe236))

print("\n=== RECORD-EFFECT NOTE EXCERPTS ===")
notes262 = Path("/work/notes.md").read_text()
matches262 = []
for needle in ("six-lag", "six lag", "impulse", "distributed lag", "kernel", "identification"):
    start = 0
    hits = []
    while True:
        p = notes262.lower().find(needle, start)
        if p < 0:
            break
        hits.append(p)
        start = p + len(needle)
    if hits:
        p = hits[-1]
        matches262.append((p, notes262[max(0, p-200):p+750]))
for p, text in sorted(matches262)[-4:]:
    print(text)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 262\n"
            "Architecture confirms core.b_in is an additive recurrent injection after the blocks. "
            "Cached factualE258 and all ten Es259 counterfactual features are available. "
            "Checking local response units, physical-rate alignment and record-effect evidence "
            "before fitting a factual-anchored rank-two edit. No weights changed. "
            "Counts: 262 Python runs, 27 forwards.\n")
```