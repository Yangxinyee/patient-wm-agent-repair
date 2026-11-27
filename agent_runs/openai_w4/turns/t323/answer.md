The effective norepinephrine output columns already oppose each other, and doubling the innovation gain exceeded the CPU fit budget. I’ll also test an asymmetric insulin response: stronger for dose increases and weaker for decreases, since the current doubling response is too small while halving is too large. Factual screening will tell us whether that correction leaves room for a modest norepinephrine gain.

```python
import re
import numpy as np
import torch
from pathlib import Path

def asymmetric_ins279(gain_up, gain_down, gate_shift=26.0,
                      ne_gain=None, ne_gate=0.0, ne_exact=False):
    ins = symmetric_ins265(1.0, gate_shift)
    c = (symmetric_ne272(ne_gain, exact_pair=ne_exact, gate_shift=ne_gate)
         if ne_gain is not None else ins)
    c = {n: torch.as_tensor(v).clone() for n, v in c.items()}
    wi = a262(ins["z_post.0.weight"])
    bi = a262(ins["z_post.0.bias"])
    gw = (wi[96] + wi[184]) / 2
    gb = float((bi[96] + bi[184]) / 2)
    qw = a262(ins_dw257)[96]
    qb = float(a262(ins_gain_db257)[96])
    w = a262(c["z_post.0.weight"]).copy()
    b = a262(c["z_post.0.bias"]).copy()
    w[96], w[184] = gw + gain_up * qw / 2, gw - gain_down * qw / 2
    b[96], b[184] = gb + gain_up * qb / 2, gb - gain_down * qb / 2
    c["z_post.0.weight"] = like258(w, c["z_post.0.weight"])
    c["z_post.0.bias"] = like258(b, c["z_post.0.bias"])
    return c

snapshots279 = [
    (name, obj, obj.copy())
    for name, obj in list(globals().items())
    if isinstance(obj, dict) and "core.b_in.weight" in obj
]
candidates279 = {}
results279 = {}
try:
    for key in [
        (2.0, 1.0, 26.0, None),
        (2.25, .75, 26.0, None),
        (2.5, .5, 26.0, None),
        (2.0, 1.0, 24.0, None),
        (2.25, .75, 28.0, None),
        (2.25, .75, 26.0, 1.4),
    ]:
        cc = asymmetric_ins279(*key[:3], ne_gain=key[3])
        rr = factual_eval194(full273(cc))
        nn = float(torch.as_tensor(rr[1], dtype=torch.float32).mean())
        candidates279[key] = cc
        results279[key] = {"nll": nn, "delta": nn - orig_nll277}
        print("ASYMMETRIC SCREEN", key, results279[key], flush=True)
finally:
    for name, obj, contents in snapshots279:
        obj.clear()
        obj.update(contents)
        globals()[name] = obj

print("\nEARLIER CALIBRATION NOTES")
sections279 = re.split(r"\n(?=## )", Path("/work/notes.md").read_text())
for marker in ("Run 183", "Run 186", "Run 206", "Run 223", "Run 250"):
    matches = [s for s in sections279 if marker in s.split("\n", 1)[0]]
    if matches:
        print(matches[-1][:1300])

print("\nACTION HELPER NAMES")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and any(
        s in k.lower() for s in ("embed", "atom", "action", "fixture", "decoder")
    ):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount])
print("NE RECORD COLUMNS", ne_records188.columns.tolist())
print(ne_records188.head(2).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 279\n"
            "Unsaved asymmetric insulin innovation screen (increase/decrease gains), "
            "including modest NE gain: " + repr(results279) +
            ". All model dictionaries restored; submitted tensors unchanged. "
            "Asymmetry is a rollout calibration hypothesis, not new ICU-effect evidence.\n")
```