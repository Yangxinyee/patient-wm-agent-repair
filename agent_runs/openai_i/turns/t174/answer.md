The vocabulary confirms that the tool’s mean excludes the absent class without renormalizing. The comparison failed because the states and factual outputs came from different stays, not because the readout was wrong. I’ll find matching artifacts and compare predictions immediately after drug bundles. A close match would justify proceeding to response calibration.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== FIND MATCHING FACTUAL OUTPUTS ===")
state_stays157 = set(map(int, stays156))
choices157 = []
for p157 in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    d157 = pd.read_csv(p157)
    overlap157 = len(set(map(int, d157.stay_id)) & state_stays157)
    print(str(p157), "stays", d157.stay_id.nunique(), "overlap", overlap157)
    choices157.append((overlap157, str(p157), d157))

best157 = max(choices157, key=lambda x: x[0])
if best157[0]:
    factual157 = best157[2]
    lookup157 = {
        (int(r.stay_id), round(float(r.hour), 5), str(r.event)): float(r.predicted_mean)
        for r in factual157.itertuples(index=False)
    }
    drug_mask157 = (
        archive156["has_insulin"].astype(bool)
        | archive156["has_norepinephrine"].astype(bool)
    )
    candidates157 = np.flatnonzero(drug_mask157 & (hours156 >= 6) & (hours156 <= 22))
    validation157 = {}
    for event157 in ("glucose_finger", "map_arterial"):
        by_key157 = {}
        for i157 in candidates157:
            key157 = (int(stays156[i157]),
                      round(float(np.floor(hours156[i157]) + 1), 5), event157)
            if key157 in lookup157:
                if key157 not in by_key157 or hours156[i157] > hours156[by_key157[key157]]:
                    by_key157[key157] = i157
        items157 = list(by_key157.items())
        select157 = np.linspace(0, len(items157)-1, min(64, len(items157)), dtype=int)
        items157 = [items157[j] for j in select157]
        ix157 = np.array([i for _, i in items157], dtype=int)
        target157 = np.array([lookup157[k] for k, _ in items157])
        print("\nREADOUT VALIDATION:", event157, "n =", len(ix157))
        if not len(ix157):
            continue
        h157 = torch.tensor(S[ix157, 192:960]).reshape(-1, 4, 192)
        out157 = torch.tensor(S[ix157, :192])
        validation157[event157] = []
        for ti157 in range(5):
            raw157, cond157, absent157, cap157 = readout_capture156(
                h157, out157, stays156[ix157], hours156[ix157], event157, ti157)
            err157 = raw157.numpy() - target157
            stats157 = (
                ti157, float(np.sqrt(np.mean(err157**2))),
                float(np.mean(err157)), float(absent157.mean())
            )
            validation157[event157].append(stats157)
            print("type, raw RMSE, bias, absent probability:", stats157)
        print("observed output range:", float(target157.min()), float(target157.max()))
        print("example last-register vector:", cap157["register"][0].tolist())
else:
    print("NO MATCHING FACTUAL ARTIFACT; a matching forward request is needed.")

print("\n=== TRAINING RECORD TIMING EXAMPLE ===")
sid157 = int(stays156[0])
print(records[(records.stay_id == sid157)
              & (records.hour >= 10) & (records.hour < 11.2)].to_string(index=False))

print("\n=== EARLIER RECORD-EVIDENCE NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:5200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 157 — matched factual/state validation\n"
            "- Vocabulary specifies unnormalized physical-bin mean; the absent class contributes zero.\n"
            "- Previous comparison had no overlapping stays (training states versus tuning factual output).\n"
            "- Searching retained factual artifacts for overlap and validating post-drug readouts.\n"
            "- No weights changed.\n")
```