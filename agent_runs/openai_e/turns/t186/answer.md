The insulin-only edit passes on these stays, but its +0.00464 NLL cost leaves almost no margin for norepinephrine. I’ll inspect the decoder and paired errors for a small record-based calibration that could recover factual fit. It must improve held-out NLL without erasing the dose response to be useful.

```python
import inspect, numpy as np, pandas as pd, re
from pathlib import Path

orig159 = pd.read_csv("/work/fwd_21/factual_original.csv")
edit159 = pd.read_csv("/work/fwd_21/factual_edited.csv")
pair159 = orig159.merge(edit159, on=["stay_id","hour","event","observed"],
                       suffixes=("_original","_edited"), validate="one_to_one")
pair159["delta_nll"] = pair159.nll_edited - pair159.nll_original
print("INSULIN-ONLY PAIRED NLL:")
print(pair159.groupby("event").delta_nll.agg(["mean","std","count"]))
print("DECODER TENSORS:")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value_bin", "embed_transition.value_bin"]):
        print(n, SHAPES[n])
print("HELPER INTERFACES:")
for n in ["readout121","readout125","extract_readout143","linear0","norm0","mlp0"]:
    x = globals().get(n)
    if callable(x):
        print(n, inspect.signature(x))
        if n.startswith("readout"):
            print("  defaults:", [(type(v).__name__, getattr(v,"shape",None))
                                   for v in (x.__defaults__ or ())])
            print("  referenced objects:", x.__code__.co_names)
print("DECODER DATA OBJECTS:")
for n in sorted(list(globals())):
    if re.search(r"(head_meta|head_transition|head_centres|register|static_all|readout12|prob12|hidden12|logit12|pred12|input12|q12)", n):
        x = globals()[n]
        if not callable(x):
            print(n, type(x).__name__, getattr(x,"shape",None))
            if n == "head_meta":
                print(x.head(3).to_string(index=False) if hasattr(x,"head") else repr(x)[:800])
print("RECONSTRUCTION CHECKS:")
sections159 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
for s in sections159:
    if re.search(r"^## Run (12[1-8]|10[0-3])\b", s):
        print(s[:550])
with open("/work/notes.md","a") as f:
    f.write("\n## Forward 21 / Run 159: insulin-only cost and calibration feasibility\n"
            "Insulin-only baseline77 factual NLL=1.63925 vs original=1.63461 (+.00464) "
            "on first 100 tuning stays: passes narrowly. Inspecting decoder interfaces for a small "
            "record-based calibration to recover factual margin without a model-weight optimizer. "
            "No weights changed. Python runs: 159; forwards: 21.\n")
```