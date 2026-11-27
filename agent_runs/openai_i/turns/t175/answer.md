I found factual outputs for all 100 stays represented in the saved states. The next error came from passing arrays to a helper that expects one stay and hour at a time. I’ll fix that call pattern and finish the comparison. The readout is usable only if the matched predictions agree.

```python
import numpy as np
import torch
from pathlib import Path

lookup158 = {
    (int(r.stay_id), round(float(r.hour), 5), str(r.event)): float(r.predicted_mean)
    for r in best157[2].itertuples(index=False)
}
validation158 = {}
validation_samples158 = {}

for event158 in ("glucose_finger", "map_arterial"):
    by_key158 = {}
    for i158 in candidates157:
        key158 = (int(stays156[i158]),
                  round(float(np.floor(hours156[i158]) + 1), 5), event158)
        if key158 in lookup158:
            if key158 not in by_key158 or hours156[i158] > hours156[by_key158[key158]]:
                by_key158[key158] = i158
    items158 = list(by_key158.items())
    chosen158 = np.linspace(0, len(items158)-1, min(40, len(items158)), dtype=int)
    items158 = [items158[j] for j in chosen158]
    validation_samples158[event158] = items158
    target158 = np.array([lookup158[k] for k, _ in items158])
    validation158[event158] = []
    print("\nREADOUT VALIDATION:", event158, "n =", len(items158))
    for ti158 in range(5):
        pred158, absent158 = [], []
        for key158, i158 in items158:
            raw158, conditional158, p0158, cap158 = readout_capture156(
                torch.tensor(S[i158:i158+1, 192:960]).reshape(1, 4, 192),
                torch.tensor(S[i158:i158+1, :192]),
                int(stays156[i158]), float(hours156[i158]), event158, ti158)
            pred158.append(float(raw158[0]))
            absent158.append(float(p0158[0]))
        errors158 = np.array(pred158) - target158
        stats158 = (
            ti158,
            float(np.sqrt(np.mean(errors158**2))),
            float(np.mean(errors158)),
            float(np.mean(absent158))
        )
        validation158[event158].append(stats158)
        print("type, raw RMSE, bias, absent probability:", stats158)
    print("example register:", cap158["register"][0].tolist())

print("\n=== TRAINING RECORD TIMING EXAMPLE ===")
sid158 = int(stays156[0])
print(records[(records.stay_id == sid158)
              & (records.hour >= 10) & (records.hour < 11.2)].to_string(index=False))

print("\n=== EARLIER RECORD-EVIDENCE NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:5200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 158 — scalar-register factual validation\n"
            "- Located matching original factual outputs in fwd_4 for all 100 training-state stays.\n"
            "- Python 157 passed arrays into a scalar stay/hour register helper; corrected the driver.\n"
            "- Comparing post-drug readouts with matching factual predictions; no weights changed.\n")
```