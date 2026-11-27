The anchored sensitivities were computed successfully; only the inventory failed when it encountered a Python class rather than an array. I’ll inspect the contrast metadata without recomputing them. Before fitting anything, the factor-one features must agree with the factual features and their bundle indices must align.

```python
import numpy as np, torch, pandas as pd
from pathlib import Path

snap137 = dict(globals())
print("Corrected contrast dictionaries:")
for n137, v137 in snap137.items():
    if isinstance(v137, dict):
        matching137 = []
        for k137, x137 in v137.items():
            if isinstance(x137, (np.ndarray, torch.Tensor, pd.DataFrame)) and (
                len(x137.shape) >= 1 and x137.shape[0] in [1497, 5988]
            ):
                matching137.append((k137, tuple(x137.shape)))
        if matching137:
            print(n137, matching137[:8])

print("\nBundle/contrast metadata:")
for n137, v137 in snap137.items():
    if isinstance(v137, pd.DataFrame) and len(v137) in [1497, 5988, 4356, 4751]:
        print(n137, tuple(v137.shape), list(v137.columns)[:18])
    elif isinstance(v137, (list, tuple)) and len(v137) == 1497:
        print(n137, type(v137).__name__, repr(v137[0])[:200])
    elif isinstance(v137, (np.ndarray, torch.Tensor)) and v137.ndim == 1 and (
        v137.shape[0] == 1497 and
        any(s in n137.lower() for s in ["idx", "index", "drug", "stay", "mask", "target"])
    ):
        print(n137, tuple(v137.shape), repr(v137[:4])[:120])

print("\nState metadata example:")
print(state_df.head(3).to_string(index=False))
print("\nVocabulary codes:", VOCAB["codes"])
print("Reading source range:", int(src_all121.min()), int(src_all121.max()))
print("Next-reading source coverage:",
      len(np.unique(src_all121.numpy())),
      "of", len(core_state), "bundles")
print("CF index range:", int(cf_idx46.min()), int(cf_idx46.max()))

print("\nSaved trial fields:")
for path137 in ["/work/ne_trial103.npz", "/work/baseline77.npz",
                "/work/ne_reduced107.npz"]:
    with np.load(path137) as a137:
        print(Path(path137).name,
              [(k, a137[k].shape) for k in a137.files])

print("\nRecent corrected-feature helper names:")
for n137, v137 in snap137.items():
    if callable(v137) and hasattr(v137, "__code__") and any(
        s in n137.lower() for s in ["prelude", "factor", "cf_feature", "correct", "dose"]
    ):
        print(n137, v137.__code__.co_varnames[:v137.__code__.co_argcount])

with open("/work/notes.md", "a") as f137:
    f137.write("\n## Run 137: safe contrast alignment inventory\n"
               "Run136 completed anchored readout and all 14 sensitivity directions "
               "(decoder agreement exact); only its metadata inventory failed on "
               "a class descriptor. Retained sensitivities unchanged. Recovering "
               "contrast pairing and bundle indices safely. No weights changed. "
               "Python runs: 137; forwards: 15.\n")
```