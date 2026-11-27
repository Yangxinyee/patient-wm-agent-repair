The early-bundle hypothesis has support: pre-ICU readings account for about half of the stronger edit’s total NLL increase. Later factual shifts are much smaller. I’ll recover the initial-bundle features and constrain their dose signal to zero. This should preserve the drug-dose contrasts while reducing the unrelated early prediction changes; a loss of those contrasts would refute the approach.

```python
import inspect, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== FULL FEATURE ARRAYS ===")
for k145, v145 in list(globals().items()):
    if isinstance(v145, (np.ndarray, torch.Tensor)):
        s145 = tuple(v145.shape)
        if len(s145) >= 2 and (s145[0] in [4250, 4350, 11440] or
                              ("prelude" in k145.lower()) or
                              (k145.startswith("E") and len(s145) == 3)):
            print(k145, s145, str(v145.dtype))

print("=== HELPER SIGNATURES AND DEPENDENCIES ===")
for k145 in ["embed_record_atom_v2", "encode_atom_set",
             "previous_context_repair", "encode_variable_sets_repair",
             "prelude_features111", "step104", "encode_flat86"]:
    f145 = globals().get(k145)
    if callable(f145):
        print(k145, inspect.signature(f145))
        print("dependencies:", f145.__code__.co_names)
        print("constants:", [x for x in f145.__code__.co_consts
                             if isinstance(x, (str, float, int, tuple))][:35])

print("=== CACHED NPZ METADATA ===")
for k145 in ["S_object120"]:
    if k145 in globals():
        o145 = globals()[k145]
        print(k145, o145.files)
        for key145 in o145.files:
            v145 = o145[key145]
            print(key145, v145.shape, v145.dtype,
                  v145[:3] if v145.ndim <= 1 else "")
print("=== ENCODING RELATED CACHE ===")
for k145 in ["raw_fits119", "repair_head_cfg"]:
    if k145 in globals():
        v145 = globals()[k145]
        print(k145, v145 if k145 == "repair_head_cfg" else
              {k: [(type(x).__name__, getattr(x, "shape", None)) for x in v]
               for k, v in v145.items()})
print("=== FIRST STAY EVENTS ===")
print(tuning[tuning.stay_id == tuning.stay_id.iloc[0]].head(18).to_string(index=False))
print("=== FIRST BEFORE-STATE NORMS ===")
print(np.linalg.norm(S_before119[:5], axis=1))
print("=== FORWARD DIRECTORY CONTENTS ===")
for p145 in sorted(Path("/work").glob("fwd_*")):
    print(p145.name, [x.name for x in p145.iterdir()])
```
