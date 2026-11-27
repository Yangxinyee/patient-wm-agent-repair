The architecture confirms that the edit changes recurrent dynamics, not a post-processing correction. The visible record analysis supports a negative insulin effect, but the output was truncated before the later checks. I’ll retrieve the final effect estimates and exact rollout IDs so the next comparison tests the stronger edit against evidence from the records.

```python
from pathlib import Path
import re
import pandas as pd

dose38_path243 = Path("/work/fwd_38/dose_edit_edited.csv")
dose38_df243 = pd.read_csv(dose38_path243)
print("MATCHED ROLLOUT FILE")
print("Columns:", list(dose38_df243.columns))
print(dose38_df243.head(1).to_string(index=False))
stay_col243 = next(c for c in dose38_df243.columns if "stay" in c.lower())
dose48_ids = sorted(dose38_df243[stay_col243].dropna().astype(int).unique().tolist())
assert 0 < len(dose48_ids) <= 100
print("MATCHED STAY IDS:", dose48_ids)

notes243 = Path("/work/notes.md").read_text()
heading243 = ""
matches243 = []
for line in notes243.splitlines():
    if line.startswith("##"):
        heading243 = line
    if re.search(
        r"target|latent persistence|causal estimate|identification assumption|"
        r"effect estimate|rho|physical effect|record-derived|record-based.*slope",
        line, re.I
    ):
        matches243.append(heading243 + "\n" + line[:650])

print("\nCOMPACT CAUSAL-EVIDENCE EXCERPTS")
text243 = "\n".join(dict.fromkeys(matches243))
if len(text243) > 5300:
    print(text243[:3500])
    print("\n[Middle omitted]\n")
    print(text243[-1800:])
else:
    print(text243)

print("\nRELEVANT PERSISTENT VARIABLES")
print([k for k in keys242
       if re.search(r"target|causal|slope|persist|rollout|dose.*ids", k, re.I)])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run243: compact causal-target and rollout recovery\n"
        f"Recovered {len(dose48_ids)} matched rollout stays from forward38. "
        "No weights changed. Next compare candidate_B237 on these stays "
        "using the exact dose-edit rollout; do not treat model predictions "
        "or observational effect estimates as revealed simulator truth.\n"
    )
```