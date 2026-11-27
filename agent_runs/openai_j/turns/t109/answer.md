The audit rules out stale edited weights, but the posterior fits do not generalize reliably across dose factors. Before another edit, I will recover the architecture and record-based effect estimates, then resolve the CPU/GPU prediction mismatch. Matching individual GPU predictions—not just average NLL—will be the validation criterion.

```python
from pathlib import Path
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

notes = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes[:-8000][-20000:])

for filename in [
    "dml_results.csv", "dynamic_iv.csv", "proxy_iv.csv",
    "sensor_iv.csv", "local_projections.csv", "kernel_targets.csv"
]:
    path = Path("/work") / filename
    table = pd.read_csv(path)
    print(f"\n=== {filename}: {table.shape} ===")
    print(table.to_string(index=False, max_rows=45))

print("\n=== RECONSTRUCTION HELPERS ===")
for name, value in list(globals().items()):
    if callable(value) and hasattr(value, "__code__"):
        if any(s in name.lower() for s in [
            "repair", "readout", "head", "numeric", "context",
            "bundle", "layout", "predict", "reconstruct"
        ]):
            c = value.__code__
            print(name, "arguments:", c.co_varnames[:c.co_argcount],
                  "references:", c.co_names)

print("\n=== DATAFRAME CACHE ===")
for name, value in list(globals().items()):
    if isinstance(value, pd.DataFrame):
        print(name, value.shape, list(value.columns))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Restart after run 95\n"
        "- Cached tensors match original weights; no active edits.\n"
        "- Posterior contrast fits show poor extrapolation across dose factors.\n"
        "- Recovering architecture and record-derived causal evidence; require "
        "row-level CPU/GPU prediction agreement before another readout edit.\n"
        "- Forward passes used: 12.\n"
    )
```