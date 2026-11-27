The vocabulary supplies centres only for the 16 physical-value bins. I’ll separate the absent class and compare both mean conventions with the original factual outputs, also checking the event-type index. Close agreement would validate the readout; large errors would mean its other inputs still need correction.

```python
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

torch.set_num_threads(4)

# Give the existing helper a zero contribution for the absent class.
# Separately compute the mean conditional on a physical value below.
if "centres16_156" not in globals():
    centres16_156 = centres153

def centres153(event):
    c = centres16_156(event)
    return torch.cat([torch.zeros(1, dtype=c.dtype), c])

def readout_capture156(h, out, stay, hour, event, type_index):
    captured = {}
    code = readout153.__code__
    old_profile = sys.getprofile()
    def profile(frame, action, arg):
        if action == "return" and frame.f_code is code:
            for key in ("logits", "register", "last_bin", "ctx", "with_code"):
                if key in frame.f_locals:
                    captured[key] = frame.f_locals[key]
    try:
        sys.setprofile(profile)
        raw = readout153(h, out, stay, hour, event, type_index)
    finally:
        sys.setprofile(old_profile)
    logits = captured["logits"]
    p = torch.softmax(logits, dim=-1)
    conditional = torch.softmax(logits[:, 1:], dim=-1) @ centres16_156(event)
    return raw, conditional, p[:, 0], captured

archive156 = np.load("/work/fwd_3/states_original.npz", allow_pickle=False)
print("STATE FIELDS:", [(k, archive156[k].shape) for k in archive156.files])
print("EVENT KINDS:", VOCAB["event_kinds"])
print("VALUE-BIN NOTE:", VOCAB.get("value_bin_note"))

factual156 = pd.read_csv("/work/fwd_1/factual_original.csv")
print("FACTUAL COLUMNS:", list(factual156.columns))
print(factual156.head(2).to_string(index=False))

nsk156 = [k for k in archive156.files
          if "stay" in k.lower() and archive156[k].shape == (len(S),)]
nhk156 = [k for k in archive156.files
          if "hour" in k.lower() and archive156[k].shape == (len(S),)]
mean_cols156 = [c for c in factual156.columns
               if "mean" in c.lower() and ("pred" in c.lower() or c.lower() == "mean")]
if nsk156 and nhk156 and mean_cols156:
    stays156 = archive156[nsk156[0]]
    hours156 = archive156[nhk156[0]]
    mean_col156 = mean_cols156[0]
    factual_lookup156 = {
        (int(r.stay_id), round(float(r.hour), 5), str(r.event)): float(getattr(r, mean_col156))
        for r in factual156.itertuples(index=False)
    }
    candidates156 = np.flatnonzero(
        (hours156 >= 6) & (hours156 <= 22)
        & (np.abs(hours156 - np.round(hours156)) > 0.02)
    )
    for event156 in ("glucose_finger", "map_arterial"):
        matched156 = []
        for i156 in candidates156:
            next_hour156 = float(np.floor(hours156[i156]) + 1)
            key156 = (int(stays156[i156]), round(next_hour156, 5), event156)
            if key156 in factual_lookup156 and key156[:2] not in {
                x[2][:2] for x in matched156
            }:
                matched156.append((i156, factual_lookup156[key156], key156))
        if not matched156:
            print(event156, "NO MATCHED FACTUAL READINGS")
            continue
        pick156 = np.linspace(0, len(matched156) - 1,
                              min(48, len(matched156)), dtype=int)
        matched156 = [matched156[j] for j in pick156]
        ix156 = np.array([x[0] for x in matched156])
        target156 = np.array([x[1] for x in matched156])
        print("\nREADOUT CHECK:", event156, "n =", len(ix156))
        for ti156 in range(5):
            raw156, conditional156, absent156, cap156 = readout_capture156(
                torch.tensor(S[ix156, 192:960]).reshape(-1, 4, 192),
                torch.tensor(S[ix156, :192]),
                stays156[ix156], hours156[ix156], event156, ti156)
            raw_np156 = raw156.numpy()
            cond_np156 = conditional156.numpy()
            print("type", ti156,
                  "raw RMSE", round(float(np.sqrt(np.mean((raw_np156-target156)**2))), 4),
                  "conditional RMSE", round(float(np.sqrt(np.mean((cond_np156-target156)**2))), 4),
                  "conditional bias", round(float(np.mean(cond_np156-target156)), 4),
                  "absent mean", round(float(absent156.mean()), 6))
        print("example register:", cap156["register"][0].tolist())
else:
    print("Validation metadata unresolved:", nsk156, nhk156, mean_cols156)

print("ACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 156 — repaired readout validation\n"
            "- Separated the absent value class from 16 physical centres.\n"
            "- Comparing raw and conditional means and event-type choices against original factual outputs.\n"
            "- No weights changed; response directions require this validation.\n")
```