The query-only edit harmed factual fit without improving the dose responses, so removing it was justified. The smaller recurrent-gain edit is still untested. I’ll recover the probe stays and earlier causal evidence before checking its factual loss. It remains viable only if the NLL increase stays below 0.005 and its dose responses move toward the record-based estimates.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== CURRENT OPERATIONS ===")
print(ops())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK / RECORD-BASED EVIDENCE ===")
print(notes_text[:16000])

print("\n=== RECOVERED PROBE AND LAST RESULTS ===")
for k, kind in [(41, "factual"), (42, "dose_edit")]:
    for model in ["original", "edited"]:
        p = Path(f"/work/fwd_{k}/{kind}_{model}.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        print(f"\nForward {k}, {kind}, {model}: {len(df)} rows")
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
        if "stay_id" in df:
            ids = sorted(df["stay_id"].unique().tolist())
            print("Stay IDs:", ids)
            if k == 41 and model == "original":
                recovered_probe_ids = ids
        if "nll" in df:
            print("Overall NLL:", df["nll"].mean())
            print(df.groupby("event")["nll"].agg(["mean", "count"]))

print("\n=== RELEVANT SAVED VARIABLES ===")
for name, value in list(globals().items()):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in
           ["target", "causal", "effect", "probe", "record_rho", "record_beta"]):
        if isinstance(value, np.ndarray):
            print(name, "array", value.shape)
        elif isinstance(value, pd.DataFrame):
            print(name, "DataFrame", value.shape, value.columns.tolist()[:16])
        elif isinstance(value, (dict, list, tuple, str, int, float)):
            text = repr(value)
            print(name, text[:1800])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 111: restart recovery\n"
        "- Retained numeric gain 4 and recurrent input gain 1.35; neither operation changed.\n"
        "- Recovered architecture, probe stays, and earlier record-based evidence.\n"
        "- Current candidate remains unvalidated; next test is factual NLL.\n"
        "- Forward passes used: 42; 8 remain.\n"
    )
```