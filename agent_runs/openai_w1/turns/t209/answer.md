The dose branch responds in roughly the expected directions, but its insulin residual is not especially small on factual bundles. I’ll also test a simpler route: fit a value-head correction from the paired dose-sensitive hidden features. It will be constrained to preserve factual logits and validated on separate stays before any installation.

```python
import numpy as np
import pandas as pd
import torch

print("PAIRED HEAD KEYS")
print("original", list(hp_original_head))
print("probe", list(hp_probe_head))
print("rows columns", list(hp_original_head["rows"].columns))
print(hp_original_head["rows"].head(3).to_string(index=False))

f187 = np185(hp_phi0).astype(np.float64)
d187 = np185(hp_dphi).astype(np.float64)
s187 = np185(hp_sensitivity)
print("\nCONTRAST SCALES",
      "factual RMS", float(np.sqrt(np.mean(f187**2))),
      "contrast RMS", float(np.sqrt(np.mean(d187**2))),
      "largest generalized sensitivities", s187[-12:].tolist())
print("he_probe_scale", he_probe_scale)
num187 = np185(he_numeric_original)
probe187 = np185(he_numeric_probe)
for j187 in [2,4]:
    print("numeric column", j187,
          "scale", float(np.sum(probe187[:,j187]*num187[:,j187]) /
                         np.sum(num187[:,j187]**2)))
print("\nNUMERIC NORMALISATION")
for k187, v187 in VOCAB["numeric"].items():
    if str(k187) in ["insulin", "norepinephrine"] or "dose" in str(k187) or "rate" in str(k187):
        print(k187, str(v187)[:850])

print("\nRECORD COLUMNS", list(ds_records.columns))
print(ds_records.loc[ds_records.event.isin(["insulin","norepinephrine"])].head(3).to_string(index=False))
print("\nPAIR ALIGNMENT")
print("rows equal", hp_original_head["rows"].equals(hp_probe_head["rows"]))
print("phi equal sizes", f187.shape, d187.shape)
print("factual head means", float(np.mean(np185(hp_mean0))),
      "probe mean change", float(np.mean(np185(hp_meanp)-np185(hp_mean0))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 187\nTwo-coordinate carrier remains possible, but direct value-head "
            "contrast regression is simpler and leaves all non-value heads unchanged. "
            "Checking matched features and dose-probe semantics before a factual-zero anchored fit. No edits.\n")
```