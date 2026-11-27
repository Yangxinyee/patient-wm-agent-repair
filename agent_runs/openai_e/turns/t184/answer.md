The architecture confirms that dose edits enter the recurrent state. Earlier record analyses suggest a negative insulin effect, but the norepinephrine estimate was still provisional. The output was truncated before the validation IDs, so I’ll retrieve those and the later evidence before testing factual fit.

```python
import pandas as pd, numpy as np, re
from pathlib import Path

f158 = pd.read_csv("/work/fwd_20/factual_original.csv")
print("VALIDATION IDS:", sorted(f158["stay_id"].unique().astype(int).tolist()))
print("FACTUAL COLUMNS:", f158.columns.tolist())
print("CURRENT EDITS:", edited())

sections158 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("=== LATER RECORD-BASED EFFECT EVIDENCE ===")
evidence158 = [
    s for s in sections158
    if re.search(r"\b(RD|DML|causal|identification|true effect|target effect)\b", s, re.I)
    and not re.search(r"^## (Initial|Run [1-9]:|Run 15[2-7]:)", s)
]
for s in evidence158[-5:]:
    print(s[:1100])
print("=== INSULIN BASELINE VALIDATION HISTORY ===")
for s in sections158:
    if re.search(r"^## (Run (6[3-9]|7[0-9])\b|Forward (7|8|9|10)\b)", s):
        print(s[:700])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 158: bounded recovery\n"
            "Recovered factual-test stay IDs and selected later causal/baseline evidence after truncated output. "
            "No weights changed. Python runs: 158; forwards: 20.\n")
```