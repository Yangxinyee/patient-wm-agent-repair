The conservative five-tensor edit is restored. The failed candidate uses large dose-decoding weights, which may amplify feature variation not covered by the small fitting sample. I’ll recover the factual feature-building pipeline and saved state files so I can broaden the distillation sample. Agreement on independent stays would support that diagnosis.

```python
import os, inspect, numpy as np
from pathlib import Path

print("=== SAVED STATE FILES ===")
for p221 in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(p221) as z221:
        print(str(p221), {k: z221[k].shape for k in z221.files})

print("\n=== FEATURE-BUILDING INTERFACES ===")
for n221 in ["contexts37", "context64", "register67", "decode67", "ob53",
             "head_inputs96", "f176", "f177", "norm63", "linear63"]:
    f221 = globals()[n221]
    print(n221, inspect.signature(f221))
    print(" locals:", f221.__code__.co_varnames)
    print(" references:", f221.__code__.co_names)

print("\n=== READING-SIZED CACHE ARRAYS ===")
for n221, v221 in sorted(list(globals().items())):
    if isinstance(v221, (np.ndarray, torch.Tensor)) and (
        4828 in tuple(v221.shape) or 4491 in tuple(v221.shape)):
        if not re.search(r"(19[0-9]|20[0-9]|21[0-9])$", n221):
            print(n221, tuple(v221.shape))
print("\n=== SMALL DATAFRAMES ===")
for n221, v221 in sorted(list(globals().items())):
    if isinstance(v221, pd.DataFrame) and len(v221) < 15000:
        print(n221, v221.shape, list(v221.columns)[:14])

print("\n=== FEATURE-CONSTRUCTION NOTEBOOK ENTRIES ===")
for s221 in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.search(r"^## .*?(?:Run 3[6-7]\b|Run 6[5-7]\b|Run 17[1-3]\b)", s221):
        print(s221[:850])
```
