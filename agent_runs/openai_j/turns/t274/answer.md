The candidate passed cached factual checks, but its rollout effects and GPU factual loss remain unverified. I’ll recover the architecture, earlier causal evidence, and exact stay IDs before testing it. A factual-loss increase above 0.005 would reject it; better dose responses within that limit would support keeping it.

```python
from pathlib import Path
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

audit244_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE ===")
print(audit244_notes[:8500])

audit244_tuning = pd.read_csv("/data/tuning.csv", nrows=8)
print("\n=== TUNING COLUMNS AND SAMPLE ===")
print(audit244_tuning.to_string(index=False))

print("\n=== RECOVERY SYMBOLS ===")
print(sorted(k for k in globals()
             if not k.startswith("_")
             and any(s in k.lower() for s in
                     ["243", "242", "head177", "map192", "kernel", "target",
                      "tune", "stay", "install", "causal", "iv", "contrast"])))
print("\n=== ACTIVE EDITS ===")
print(edited())

with open("/work/notes.md", "a") as audit244_f:
    audit244_f.write(
        "\n## Run 244: restart audit\n"
        "- Independent-strength candidate remains installed; no weights changed.\n"
        "- Recovering architecture, record-based causal evidence, and exact stay IDs before GPU validation.\n"
        "- Forward passes used: 24; Python runs used: 244.\n"
    )
```