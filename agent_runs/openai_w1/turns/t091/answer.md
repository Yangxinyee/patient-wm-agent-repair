The cached bundles are correctly aligned, and END-only states reconstruct exactly. The remaining error is concentrated in reading bundles, so calibration based on those reconstructions is not yet trustworthy. I’ll check the architecture, vocabulary and actual measurement fields next. Correct field encoding should eliminate the posterior mismatch; if it does not, the encoder implementation remains suspect.

```python
from pathlib import Path
import json
import pandas as pd

r82_arch = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(r82_arch)

r82_vocab = json.loads(Path("/model/vocab.json").read_text())
print("\n=== VOCAB TOP-LEVEL KEYS ===")
print(list(r82_vocab))

def r82_show_fields(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k) in {
                "unit", "rate_unit", "categorical", "categories",
                "normalization", "normalisation", "numeric_fields"
            }:
                text = json.dumps(v)
                print(p, text[:3500])
            elif isinstance(v, dict):
                r82_show_fields(v, p)

print("\n=== FIELD DEFINITIONS ===")
r82_show_fields(r82_vocab)

print("\n=== RECORD DOCUMENTATION ===")
print(Path("/data/README.md").read_text()[:6500])

r82_sample = pd.read_csv("/data/train.csv", nrows=40)
print("\n=== ACTUAL RECORD FIELDS ===")
print(r82_sample.columns.tolist())
print(r82_sample.to_string(index=False, max_rows=16))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 82\n"
        "Restart recovery: verified prior evidence of exact bundle alignment and "
        "END-only posterior agreement. Reading posterior reconstruction is still "
        "invalid; checking architecture, vocabulary and actual categorical fields "
        "before calibration. No weights changed. "
        "Counts: 82 Python runs, 6 forward passes.\n"
    )
```