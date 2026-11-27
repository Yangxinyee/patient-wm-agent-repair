Opening the gate by 10–20 units was safe on CPU; larger openings exceeded the factual-fit limit. I’ll leave the gate conservative and test stronger action-innovation gains, with and without exact opposing output columns. This will distinguish an underpowered dose signal from a useful component accidentally removed by symmetrization.

```python
import numpy as np
import torch
from pathlib import Path

print("EFFECTIVE NE OUTPUT DIRECTIONS")
for label, cc in [("prior", backup275), ("exact_pair", candidate275)]:
    zz = a262(cc["z_in.weight"])
    oo = a262(cc["z_post.2.weight"])
    vv = zz @ oo[:32, [156, 89]]
    print(label, "column norms", np.linalg.norm(vv, axis=0).tolist(),
          "common-mode norm", float(np.linalg.norm(vv.sum(axis=1))),
          "contrast norm", float(np.linalg.norm(vv[:, 0] - vv[:, 1])))

snapshots278 = [
    (name, obj, obj.copy())
    for name, obj in list(globals().items())
    if isinstance(obj, dict) and "core.b_in.weight" in obj
]
ne_candidates278 = {}
ne_results278 = {}
try:
    for gain, exact, gs in [
        (1.0, False, 0.0),
        (2.0, False, 0.0),
        (4.0, False, 0.0),
        (8.0, False, 0.0),
        (2.0, True, 0.0),
        (4.0, True, 0.0),
        (4.0, False, -20.0),
    ]:
        cc = symmetric_ne272(gain, exact_pair=exact, gate_shift=gs)
        rr = factual_eval194(full273(cc))
        nn = float(torch.as_tensor(rr[1], dtype=torch.float32).mean())
        key = (gain, exact, gs)
        ne_candidates278[key] = cc
        ne_results278[key] = {"nll": nn, "delta": nn - orig_nll277}
        print("NE GAIN SCREEN", key, ne_results278[key], flush=True)
finally:
    for name, obj, contents in snapshots278:
        obj.clear()
        obj.update(contents)
        globals()[name] = obj

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 278\n"
            "Unsaved NE gain/output-pair screen: " + repr(ne_results278) +
            ". Tested transmission hypotheses only; ICU effect estimates unchanged. "
            "Global model dictionaries restored; no submitted weights changed.\n")
```