```python
from pathlib import Path
import numpy as np
import pandas as pd
import re
import json

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-IDENTIFICATION NOTES")
print(notes_text[:6500])

print("\nSELECTED IDENTIFICATION EXCERPTS")
terms = ["instrument", "randomiz", "innovation", "impulse targets",
         "causal", "identification"]
shown = set()
for term in terms:
    matches = list(re.finditer(re.escape(term), notes_text, re.I))
    for match in matches[:2]:
        lo = max(0, notes_text.rfind("\n", 0, max(0, match.start() - 250)))
        hi = min(len(notes_text), match.end() + 650)
        if any(abs(lo - old) < 300 for old in shown):
            continue
        shown.add(lo)
        print(f"\n[{term}]")
        print(notes_text[lo:hi])

print("\nCANDIDATE ARCHIVE CONTRACTS")
for filename in ["joint274.npz", "joint284.npz", "joint291.npz"]:
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        summary = {}
        for key in archive.files:
            value = archive[key]
            summary[key] = (
                value.tolist() if value.size <= 8
                else {"shape": list(value.shape), "dtype": str(value.dtype)}
            )
        print(filename, summary)

factual48 = pd.read_csv("/work/fwd_48/factual_edited.csv")
final_rollout_stays = sorted(
    int(x) for x in factual48["stay_id"].unique()
)
assert len(final_rollout_stays) == 100
print("\nFINAL ROLLOUT STAYS")
print(json.dumps(final_rollout_stays))
print("Currently edited tensors:", edited())

with Path("/work/notes.md").open("a") as handle:
    handle.write(
        "\n## Run 294: final-rollout preparation\n"
        "Recovered architecture, identification notes, candidate archive contracts "
        "and the exact forward48 cohort. No weights changed. "
        "C remains provisional pending the final paired dose rollout. "
        "Python runs: 294; forwards: 49.\n"
    )
```