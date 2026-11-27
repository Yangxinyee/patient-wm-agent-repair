The records show that treatment associations change substantially with history controls and sensor choice. That makes the later identification checks important; the preliminary coefficients alone are not a safe repair target. I will recover those checks next. Agreement across independent sensors and lag estimates would support a shared drug-effect target; disagreement would weaken it.

```python
from pathlib import Path
import pandas as pd

for filename in [
    "kernel_targets.csv", "dynamic_iv.csv", "proxy_iv.csv",
    "sensor_iv.csv", "local_projections.csv", "dml_results.csv"
]:
    table = pd.read_csv(Path("/work") / filename)
    print(f"\n=== {filename}: {table.shape} ===")
    print("columns:", list(table.columns))
    print(table.head(9).to_string(index=False, max_colwidth=24))

notes = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes.splitlines() if line.startswith("##")))

print("\n=== IDENTIFICATION NOTES ===")
start = notes.find("## Factual baseline")
end = notes.find("## Run 57")
section = notes[start:end if end >= 0 else start + 6000]
keywords = [
    "instrument", "causal", "effect", "impulse", "kernel", "placebo",
    "identif", "innovation", "confound", "sensor", "target", "rho",
    "beta", "rejected", "accepted"
]
lines = section.splitlines()
selected = [
    line for line in lines
    if line.startswith("##") or any(k in line.lower() for k in keywords)
]
print("\n".join(selected)[:3300])
print("\nACTIVE EDITS:", edited())
```