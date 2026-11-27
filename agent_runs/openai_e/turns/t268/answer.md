The baseline’s factual cost varies appreciably between samples. Before spending its apparent margin on norepinephrine, I need to separate that variation from trial219’s added cost. I’ll recover the fresh-sample IDs and earlier dose results, then test the restored baseline on exactly the sample that rejected the trial.

```python
from pathlib import Path
import re
import pandas as pd
import numpy as np

fresh_ids232 = sorted(
    pd.read_csv("/work/fwd_26/factual_original.csv")["stay_id"]
    .astype(np.int64).unique().tolist()
)
print("Exact fresh factual sample (100 stays):")
print(fresh_ids232)

print("\nDose-output schema:")
dose232 = pd.read_csv("/work/fwd_25/dose_edit_edited.csv")
print(list(dose232.columns))
print(dose232.head(3).to_string(index=False))

print("\nEarlier recurrent-edit findings:")
lines232 = Path("/work/notes.md").read_text().splitlines()
wanted232 = [77, 82, 83, 103, 104, 107, 108, 109, 110, 111, 112,
             148, 151, 154, 155, 156, 158]
chunks232 = []
for number in wanted232:
    pattern = re.compile(rf"\bRun\s+{number}\b", re.I)
    matches = [i for i, line in enumerate(lines232)
               if line.startswith("##") and pattern.search(line)]
    for i in matches[:1]:
        chunks232.append("\n".join(lines232[i:i+4])[:600])
print("\n---\n".join(chunks232)[:8000])

print("\nCurrent baseline delta norm:",
      float(np.linalg.norm(dbase231)))
print("Weights unchanged since baseline restoration.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 232: prepare exact fresh-sample baseline comparison\n"
        "Recovered the 100 stays from Forward26. Trial219 and baseline77 "
        "differ by only .000045 NLL on the first 100 tuning stays, so the "
        "fresh-sample excess cannot yet be attributed to balancing. "
        "Baseline77 remains installed; exact paired baseline validation next.\n"
        "Python runs: 232; forwards: 26.\n"
    )
```