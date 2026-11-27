The latest batch’s value-NLL increase was about 0.00282 nats, below the 0.005 limit. That supports the safety-margin candidate, but one batch is not enough. I’ll recover the exact validation results and saved candidates before testing further. Pooled factual fit below the limit and dose responses closer to the record-based estimates would support keeping this edit.

```python
import os, json, numpy as np, pandas as pd

arch_recovery = open("/model/ARCHITECTURE.md").read()
print("ARCHITECTURE:\n", arch_recovery[:9500])
print("Architecture characters:", len(arch_recovery))

final_frames = {}
for k in (37, 38):
    final_frames[k] = {}
    for model in ("original", "edited"):
        p = f"/work/fwd_{k}/factual_{model}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            final_frames[k][model] = d
            nc = [c for c in d.columns if "nll" in c.lower()]
            print("FWD", k, model, "rows", len(d), "columns", list(d.columns),
                  "NLL means", {c: float(d[c].mean()) for c in nc})
    if len(final_frames[k]) == 2:
        a, b = final_frames[k]["original"], final_frames[k]["edited"]
        for c in a.columns:
            if "nll" in c.lower() and c in b and len(a) == len(b):
                print("FWD", k, "paired delta", c, float((b[c]-a[c]).mean()))

for fn in ("margin_candidate.npz", "balanced_validated400.npz",
           "validated_current400.npz", "map_regularized_trial.npz"):
    z = np.load("/work/" + fn, allow_pickle=False)
    print("ARCHIVE", fn, {k: z[k].shape for k in z.files})
    for k in z.files:
        v = z[k]
        if v.size <= 8:
            print(" ", k, v)

print("Tuning header:", list(pd.read_csv("/data/tuning.csv", nrows=2).columns))
recovery_names = [k for k in list(globals()) if not k.startswith("_") and
                  any(s in k.lower() for s in ("target", "effect", "batch", "ids",
                                              "route", "candidate", "coef", "fisher"))]
print("Retained relevant variables:", recovery_names[:100])
print("Current edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 289: final-stage recovery\n"
            "- Forward 38 safety-margin candidate: value-NLL delta approximately +0.00282 on 4664 readings.\n"
            "- Recovering architecture, exact forwards 37/38, validation batches and archived components.\n"
            "- No weights changed; margin candidate still requires pooled factual and rollout validation.\n"
            "- Python runs: 289; forward passes: 38.\n")
```