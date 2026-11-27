The recovered notes suggest a much larger insulin effect than the original model predicts, but those estimates are provisional. The output was cut off before the later validation results. I’ll retrieve only the key candidate summaries and saved-edit details next; they should show whether norepinephrine needs a correction and how much factual-fit margin remains.

```python
import pandas as pd
from pathlib import Path
import re, numpy as np

sections186 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
wanted186 = [58, 63, 77, 82, 83, 103, 107, 148, 151, 154, 155, 160]
print("SELECTED EXPERIMENT SUMMARIES")
for run186 in wanted186:
    found186 = [s for s in sections186
                if re.match(rf"## Run {run186}(?:\D|$)", s)]
    for s186 in found186:
        print(s186[:480].strip())

print("\nSAVED EDIT ARRAYS")
for fn186 in ["baseline77.npz", "ne_candidate148.npz",
              "ne_candidate151.npz", "ne_candidate154.npz",
              "ne_candidate155.npz"]:
    with np.load("/work/" + fn186, allow_pickle=False) as a186:
        print(fn186, [(k, a186[k].shape) for k in a186.files])
        for k186 in a186.files:
            if a186[k186].size <= 8:
                print(" ", k186, a186[k186])

print("\nFORWARD FILE INVENTORY")
for d186 in sorted(Path("/work").glob("fwd_*"),
                   key=lambda p: int(p.name.split("_")[1])):
    print(d186.name, [p.name for p in d186.iterdir()])

print("\nCSV SCHEMAS")
for pattern186 in ["fwd_1/factual*.csv", "fwd_2/dose_edit*.csv"]:
    for p186 in Path("/work").glob(pattern186):
        x186 = pd.read_csv(p186, nrows=2)
        print(str(p186), list(x186.columns))
        print(x186.to_string(index=False))

with open("/work/notes.md", "a") as f186:
    f186.write("\n## Run 186: bounded validation and archive recovery\n"
               "Run185 output was truncated; retrieved only key experiment summaries, "
               "saved-edit array schemas, and forward-output schemas. No weights changed. "
               "Python runs: 186; forwards: 22.\n")
```