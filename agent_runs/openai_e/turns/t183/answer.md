The last run restored the insulin-only edit exactly, but did not test its factual fit. I’ll recover the model specification, record-based effect estimates, and fixed validation IDs. The next factual test will show whether insulin alone meets the NLL limit or needs revision.

```python
from pathlib import Path
import json, re
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())
notes157 = Path("/work/notes.md").read_text()
print("=== EARLY CAUSAL EVIDENCE ===")
print(notes157[:14000])
print("=== FIXED VALIDATION IDS ===")
print(first_records56["stay_id"].astype(int).tolist() if "stay_id" in first_records56.columns
      else list(first_records56.index))
print("=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 157: restart recovery before insulin-only validation\n"
            "Read architecture and record definitions; recovered early causal evidence and fixed validation IDs. "
            "Insulin-only baseline77 remains installed; no new validation or weight changes. "
            "Python runs: 157; forwards: 20.\n")
```