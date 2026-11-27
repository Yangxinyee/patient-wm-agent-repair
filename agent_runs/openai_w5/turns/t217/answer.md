With the refreshed cache, the large discontinuities disappear and dose transport is approximately linear. That weakens the routing-failure hypothesis. I’ll now inspect the installed correction and its centering terms: if those terms change with the edited dose, they may cancel part of the intended effect. Any stronger correction will still need paired factual validation.

```python
import inspect
import re
import numpy as np
import torch
from pathlib import Path

print("=== VALUE-HEAD SHAPES AND MOST-CHANGED INPUT ROWS ===")
for name in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
             "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
    print(name, SHAPES[name])
w192 = torch.as_tensor(load("attr_heads.value_bin.0.weight"))
w0192 = torch.as_tensor(load_original("attr_heads.value_bin.0.weight"))
row_change192 = (w192 - w0192).square().sum(1).sqrt()
top192 = torch.argsort(row_change192, descending=True)[:12].numpy()
print("largest changed rows:", [(int(i), round(float(row_change192[i]), 3))
                                for i in top192])
for i in top192[:8]:
    out = torch.as_tensor(load("attr_heads.value_bin.2.weight"))[:, i]
    print("row", int(i), "input private", w192[i, idx182].numpy().round(3).tolist(),
          "bias", float(torch.as_tensor(load("attr_heads.value_bin.0.bias"))[i]),
          "output column", out.numpy().round(4).tolist())

print("=== HEAD-CALIBRATION OBJECT INVENTORY ===")
for name, obj in list(globals().items()):
    if not re.search(r"(13[5-9]|14[0-9]|15[0-5])$", name):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:12])
    elif isinstance(obj, (tuple, list)):
        print(name, type(obj).__name__, str(obj)[:230] if len(obj) < 12 else len(obj))
    elif isinstance(obj, (int, float, str)):
        print(name, repr(obj)[:120])

print("=== ATOM-COUNT CHECK ===")
for j in [3, 4, 7, 31, 57, 59, 61]:
    sid = int(s4["stay_id"][j])
    minute = int(round(float(s4["hour"][j]) * 60))
    group = groups4.get((sid, minute))
    previous = groups4.get((sid, minute - 60))
    print(j, "saved atoms", int(s4["n_atoms"][j]),
          "record rows", 0 if group is None else len(group),
          "prior-hour drugs",
          [] if previous is None else previous.loc[
              previous.event.isin(["insulin", "norepinephrine"]),
              ["event", "rate"]].to_dict("records"))

print("=== NOTEBOOK END-ATOM REFERENCES ===")
text = Path("/work/notes.md").read_text()
hits = list(re.finditer(r"\bEND\b|end.atom|end marker", text, re.I))
for m in hits[:6]:
    print(text[max(0, m.start()-120):m.end()+220].replace("\n", " "))
```
