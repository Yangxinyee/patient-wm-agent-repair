The repeated output adds no new evidence. The rejected edit caused large, highly variable shifts, so a simple offset correction is not justified. I’ll audit the cached weights and recover the posterior-path diagnostics. That will distinguish a stale-weight bug from a genuinely ineffective treatment pathway before choosing the next probe.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== CACHED ORIGINAL-WEIGHT AUDIT ===")
cache_mismatch95 = []
for name95, tensor95 in W_original.items():
    fresh95 = torch.as_tensor(np.asarray(load_original(name95)), dtype=torch.float32)
    if isinstance(tensor95, torch.Tensor) and tensor95.shape == fresh95.shape:
        max95 = float((tensor95-fresh95).abs().max())
        if max95 > 1e-7:
            cache_mismatch95.append((name95, max95))
print("Cached tensors:", len(W_original))
print("Mismatches:", cache_mismatch95)
for name95, tensor95 in (("attr_heads.value_bin.2.weight", repair_head_W0),
                         ("attr_heads.value_bin.2.bias", repair_head_b0)):
    fresh95 = torch.as_tensor(np.asarray(load_original(name95)), dtype=torch.float32)
    print(name95, "stored baseline max error:", float((tensor95-fresh95).abs().max()))

print("\n=== POSTERIOR CONTRAST VALIDATION ===")
tab95 = pd.read_csv("/work/posterior_contrast_cv.csv")
print(tab95.to_string(index=False)[:2600])

print("\n=== CANDIDATE PATH TENSORS ===")
for name95 in NAMES:
    if name95.startswith(("z_post.", "core.z_in.", "core.b_in.", "core.c_out.",
                           "core.prelude.0.")):
        print(name95, SHAPES[name95])

print("\n=== RELEVANT CACHED ARRAYS ===")
for k95, v95 in sorted(dict(globals()).items()):
    if isinstance(v95, (np.ndarray, torch.Tensor)):
        shape95 = tuple(v95.shape)
        if len(shape95) == 2 and shape95[0] > 200 and shape95[1] in (32, 192, 193, 384, 576, 992):
            print(k95, shape95, type(v95).__name__)
    elif isinstance(v95, dict) and any(s95 in k95.lower() for s95 in ("post", "latent", "contrast")):
        print(k95, "keys:", list(v95)[:16])
print("fit_post_contrast arguments:",
      fit_post_contrast.__code__.co_varnames[:fit_post_contrast.__code__.co_argcount])
print("fit_post_contrast names:", fit_post_contrast.__code__.co_names)
print("repair_core_step arguments:",
      repair_core_step.__code__.co_varnames[:repair_core_step.__code__.co_argcount])
print("repair_core_step locals:", repair_core_step.__code__.co_varnames)
print("VOCAB numeric note:", VOCAB.get("numeric_note"))
print("ACTIVE EDITS:", edited())
```