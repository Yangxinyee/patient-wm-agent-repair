The head edits use paired “dose minus predicted dose” neurons, with separate gates for the four sensors. The notes also report limited numeric input support, which may affect larger interventions. I’ll now recover the embedding conventions and forward summaries so I can test finite dose changes directly.

```python
import json, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== EMBEDDING CONVENTIONS ===")
print("Event kinds:", V["event_kinds"])
print("Atom kinds:", V["atom_kinds"])
print("Numeric note:", V["numeric_note"])
print("Numeric scales:",
      [(r["code"], r["attribute"], r["median"], r["iqr"]) for r in V["numeric"]])
print("Embedding tensors:",
      [(n, tuple(W[n].shape)) for n in NAMES if n.startswith("embed.")])

print("\n=== ENCODER CHANNEL CHANGES ===")
for n in ["encode_action.norm.weight", "encode_action.norm.bias",
          "core.c_out.weight", "core.prelude.0.attn.out.weight"]:
    d = W[n] - WO[n]
    score = d.abs() if d.ndim == 1 else d.norm(dim=1)
    ii = score.argsort(descending=True)[:12]
    print(n, [(int(i), round(float(score[i]), 5)) for i in ii])
print("Action norm private weights/bias:",
      W["encode_action.norm.weight"][priv].tolist(),
      W["encode_action.norm.bias"][priv].tolist())

print("\n=== FORWARD SUMMARIES ===")
for folder in ["fwd_16", "fwd_17", "fwd_18"]:
    js = json.loads((Path("/work") / folder / "summary.json").read_text())
    print(folder, json.dumps(js.get("results", {}))[:1700])

train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
print("\n=== RECORD SAMPLE ===")
print(tuning.head(10).to_string(index=False))
tune_ids = np.sort(tuning.stay_id.unique())
print("Tuning ID ranges:",
      [(int(tune_ids[i]), int(tune_ids[min(i+99, len(tune_ids)-1)]))
       for i in range(0, len(tune_ids), 100)])
print("Current factual columns:",
      pd.read_csv("/work/fwd_17/factual_edited.csv", nrows=1).columns.tolist())
print("Current rollout columns:",
      pd.read_csv("/work/fwd_18/dose_edit_edited.csv", nrows=1).columns.tolist())
print("State schema:", [(k, S[k].shape) for k in S.files])
```