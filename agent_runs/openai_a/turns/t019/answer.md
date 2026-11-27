The insulin sensor-contrast estimates pass the reported pre-treatment checks, but norepinephrine remains poorly identified. I’ll now check the available data helpers and encoding details, then gather states for a controlled edit. The first repair will target insulin only; a substantial rollout change with less than 0.005 nats of factual degradation would support this approach.

```python
import inspect
import json
import numpy as np
import pandas as pd

print("=== HELPERS ===")
for k, v in list(globals().items()):
    if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
for k in ["config", "codes", "event_kinds", "kinds", "categorical"]:
    if k in VOCAB:
        print(k, json.dumps(VOCAB[k])[:1600])

print("\n=== REPRESENTATION ===")
print("states stay IDs:", np.unique(ss)[:8], "n:", len(np.unique(ss)))
print("matched columns:", list(matched.columns))
print(matched.head(4).to_string())
print("history columns:", list(history.columns))
print("train IDs:", train.stay_id.drop_duplicates().head(8).tolist())
print("tuning IDs:", tuning.stay_id.drop_duplicates().head(8).tolist())
print("LP parameters:", lp_fit["x"])
print("readout baseline:")
print(dose_baseline.loc[dose_baseline.reached].groupby("edit").shift.agg(["mean", "std", "count"]))
print("factual baseline:", factual_baseline.nll.mean())

print("\n=== KEY WEIGHT SHAPES ===")
for n in NAMES:
    if n.startswith(("core.c_out", "core.flow", "z_prior", "pending", "filter",
                     "embed.code", "embed.bins", "attr_heads", "embed_transition")):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart recovery; Python 16, forward 3\n"
            "- Architecture recovered; no edits yet.\n"
            "- Insulin sensor-contrast IV and one-step adjusted regressions favor a modest negative immediate effect.\n"
            "- Insulin LP estimate: persistence about 0.877, immediate effect about -1.757; causal assumptions still need scrutiny.\n"
            "- Norepinephrine estimates are control-sensitive and its sensor instrument is weak; no firm target chosen.\n"
            "- Plan: identify a conditional-dose direction in the model and test a low-rank insulin edit against paired factual/rollout checks.\n")
```
