The new MAP route has promising local dose responses, but its full-sequence factual cost is still unknown. I’ll recover the validation batches and record-derived targets before testing it officially. It is useful only if factual NLL remains within the 0.005 limit and the rollout moves toward those targets.

```python
import os, json, numpy as np, pandas as pd

print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())

notes_text = open("/work/notes.md").read()
print("\nEARLY RECORD-IDENTIFICATION NOTES")
print(notes_text[:7500])

print("\nARCHIVES")
for fn in ["map_feature_trial.npz", "map_feature_trial_backup.npz",
           "glucose_115_candidate.npz", "recurrent_glucose_valid.npz"]:
    p = "/work/" + fn
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as a:
            print(fn, {k: (a[k].shape, str(a[k].dtype)) for k in a.files})

print("\nVALIDATION FILES")
for k in [20, 21, 24, 25]:
    folder = f"/work/fwd_{k}"
    print(folder, os.listdir(folder))
    for fn in os.listdir(folder):
        if fn.endswith(".csv"):
            df = pd.read_csv(folder + "/" + fn)
            print(fn, "shape", df.shape, "columns", df.columns.tolist())
            idcols = [c for c in df.columns if "stay" in c.lower()]
            if idcols:
                print("stay IDs:", df[idcols[0]].drop_duplicates().tolist())
            print(df.head(2).to_string(index=False))

print("\nRELEVANT KERNEL OBJECTS")
for n in sorted(list(globals())):
    if not n.startswith("_") and any(s in n.lower() for s in
        ["target", "truth", "effect", "trial", "route", "candidate", "fisher",
         "replay", "inject", "predict", "slot", "validation", "chosen"]):
        obj = globals()[n]
        if callable(obj):
            print(n, "callable")
        elif isinstance(obj, np.ndarray):
            print(n, obj.shape, str(obj.dtype))
        elif isinstance(obj, dict):
            print(n, "dict keys:", list(obj)[:16])
        elif isinstance(obj, (float, int, str, tuple)):
            print(n, repr(obj)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 278: recover official-validation inputs\n"
            "- Current MAP trial remains unvalidated; glucose-only backup is preserved.\n"
            "- Recovered architecture, record-derived targets, archived components and validation stay IDs.\n"
            "- No weights changed. Python runs: 278; forward passes: 25.\n")
```