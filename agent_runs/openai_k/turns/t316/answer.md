The notes show record-derived effects in both sensor pairs, with effects persisting beyond one hour. The output was truncated before the validation IDs appeared. I’ll retrieve only those IDs and the current candidate metadata, then check whether the local factual-cost estimate survives full-sequence evaluation.

```python
import os, numpy as np, pandas as pd

for k in [24, 25]:
    folder = f"/work/fwd_{k}"
    print("FORWARD", k, os.listdir(folder))
    for fn in os.listdir(folder):
        if fn.endswith(".csv"):
            d = pd.read_csv(folder + "/" + fn)
            print(fn, d.shape, d.columns.tolist())
            sc = next((c for c in d.columns if "stay" in c.lower()), None)
            if sc:
                print("STAYS", d[sc].drop_duplicates().tolist())
            nc = [c for c in d.columns if "nll" in c.lower()]
            print("NLL means", {c: float(d[c].mean()) for c in nc})

for fn in ["map_feature_trial.npz", "map_feature_trial_backup.npz",
           "glucose_115_candidate.npz", "recurrent_glucose_valid.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as a:
        print("ARCHIVE", fn, {k: a[k].shape for k in a.files})

names = [n for n in sorted(list(globals()))
         if not n.startswith("_") and any(s in n.lower() for s in
         ["target", "truth", "trial", "fisher", "chosen", "inject",
          "replay", "validation", "candidate", "slot"])]
print("OBJECT NAMES", names)
for n in names:
    o = globals()[n]
    if isinstance(o, (float, int, str, tuple)):
        print("META", n, repr(o)[:180])
    elif isinstance(o, dict) and any(s in n.lower() for s in ["target", "truth", "chosen", "trial"]):
        print("META", n, repr(o)[:450])

lines = open("/work/notes.md").read().splitlines()
matches = [l for l in lines if any(s in l.lower() for s in
           ["target shift", "true shift", "estimated rollout", "pooled exact",
            "target:", "targets:", "convolution", "insulin x2 target"])]
print("TARGET NOTES")
for l in (matches[:6] + matches[-6:]):
    print(l[:400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 279: bounded validation recovery\n"
            "- Run 278 output truncated before validation metadata; narrowed output.\n"
            "- Record evidence supports persistent insulin/glucose and NE/MAP effects, subject to identification assumptions.\n"
            "- No weights changed. Python runs: 279; forward passes: 25.\n")
```