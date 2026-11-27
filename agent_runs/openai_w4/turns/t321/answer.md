The prior edit was restored before the inspection stopped on a bookkeeping error. The gate has a large negative bias, making dose-dependent shutoff a plausible explanation for the failed candidate. I’ll screen progressively more open gates; a useful setting must preserve factual NLL before it earns another rollout test.

```python
import numpy as np
import torch
from pathlib import Path

print("GATE DIAGNOSTICS ON CACHED POSTERIOR INPUTS")
for label, xx in [("factual_cache", inputs88), ("counterfactual_cache", cf87)]:
    xx = a262(xx)
    if xx.ndim != 2 or xx.shape[1] != 576:
        continue
    action_mask = np.linalg.norm(xx[:, 384:], axis=1) > 1e-5
    for setting, cc in [("prior", backup275), ("symmetric", candidate275)]:
        ww = a262(cc["z_post.0.weight"])
        bb = a262(cc["z_post.0.bias"])
        aa = xx @ ww[[156, 89]].T + bb[[156, 89]]
        print(label, setting, "action rows", int(action_mask.sum()))
        for j, row in enumerate((156, 89)):
            vals = aa[action_mask, j]
            print(row, "quantiles",
                  np.quantile(vals, [0, .1, .5, .9, 1]).round(3).tolist(),
                  "fraction positive", float((vals > 0).mean()))

orig_nll277 = (
    r273_results[(1.0, False)]["nll"]
    - r273_results[(1.0, False)]["delta"]
)
snapshots277 = [
    (name, obj, obj.copy())
    for name, obj in list(globals().items())
    if isinstance(obj, dict) and "core.b_in.weight" in obj
]
ne_candidates277 = {}
ne_results277 = {}
try:
    for gs in (0.0, -10.0, -20.0, -40.0, -80.0, -120.0, -200.0):
        cc = symmetric_ne272(1.0, exact_pair=True, gate_shift=gs)
        rr = factual_eval194(full273(cc))
        nn = float(torch.as_tensor(rr[1], dtype=torch.float32).mean())
        ne_candidates277[gs] = cc
        ne_results277[gs] = {"nll": nn, "delta": nn - orig_nll277}
        print("NE GATE", gs, ne_results277[gs], flush=True)
finally:
    for name, obj, contents in snapshots277:
        obj.clear()
        obj.update(contents)
        globals()[name] = obj

print("SUBMISSION UNCHANGED", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## GPU pass 31; Runs 276–277\n"
            "Run 276 restored backup275 before a diagnostic type error; its log "
            "append was not reached. Symmetric NE gate 0 had rollout shifts "
            "insulin -11.655/+9.265, NE +.072/-.513 and was rejected. "
            "Run 277 screened unsaved gate openings with exact opposing NE columns: "
            + repr(ne_results277) +
            ". Model dictionaries restored; submitted weights remain backup275.\n")
```