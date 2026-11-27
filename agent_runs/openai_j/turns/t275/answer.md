The architecture supports keeping edits in the value head: other heads remain unchanged during teacher-forced prediction, though rollout feedback can affect them. The early estimates also show that simple dose associations are unreliable. I’ll recover the later causal evidence and validation IDs without changing weights; conflicting identification checks would argue for more conservative targets.

```python
import re
import numpy as np
from pathlib import Path
import pandas as pd

audit245_ids = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
print("FIRST 100 TUNING IDS:", audit245_ids[:100])
print("SECOND BATCH ENDPOINTS:", audit245_ids[100:103], audit245_ids[197:200])

audit245_text = Path("/work/notes.md").read_text()
audit245_sections = re.split(r"(?m)(?=^## )", audit245_text)
audit245_selected = [
    s for s in audit245_sections
    if any(w in s.lower() for w in
           ["instrument", "identification", "negative control", "placebo",
            "proxy", "kernel target", "dynamic iv", "sensor iv", "causal evidence"])
]
print("\n=== IDENTIFICATION NOTES (COMPACT) ===")
for audit245_s in audit245_selected[:14]:
    print(audit245_s[:470].strip())

print("\n=== TARGET TABLE ===")
audit245_targets = pd.read_csv("/work/kernel_targets.csv")
print(audit245_targets.head(12).to_string(index=False))

print("\n=== RELEVANT HELPERS ===")
print(sorted(k for k in globals()
             if any(s in k.lower() for s in
                    ["head177", "map192", "243", "242", "install", "contrast",
                     "kernel", "iv", "causal"])
             and not k.startswith("audit"))[:100])

print("\n=== CHECKPOINT ARRAYS ===")
for audit245_file in ["independent_candidate243.npz", "validated_gain8_backup243.npz"]:
    with np.load("/work/" + audit245_file) as audit245_ck:
        print(audit245_file, {k: audit245_ck[k].shape for k in audit245_ck.files})

with open("/work/notes.md", "a") as audit245_f:
    audit245_f.write(
        "\n## Run 245: compact evidence recovery\n"
        "- Architecture read; recovering identification notes and exact tuning batches.\n"
        "- No weights changed; independent-strength candidate awaits GPU checks.\n"
    )
```