The saved arrays contain both authoritative states and locally encoded action vectors, so we can investigate the ineffective pathway without another forward pass. I’ll inspect attention inputs and the retained repair fits next. Very low attention to the latent token would explain the pilot’s failure; appreciable attention would shift the investigation to later recurrent layers.

```python
import builtins as bi
import numpy as np
from pathlib import Path

print("=== LATER REPAIR VARIABLES ===")
r65_lines = []
for r65_key, r65_obj in bi.list(globals().items()):
    if not r65_key.startswith(("r41_", "r42_", "r45_", "r47_", "r49_",
                               "r51_", "r52_", "r53_", "r58_", "r60_", "r61_", "pilot")):
        continue
    r65_shape = getattr(r65_obj, "shape", None)
    if isinstance(r65_shape, (bi.tuple, torch.Size)):
        r65_lines.append(f"{r65_key}: {bi.type(r65_obj).__name__}{bi.tuple(r65_shape)}")
    elif isinstance(r65_obj, bi.dict):
        r65_lines.append(f"{r65_key}: dict keys={bi.list(r65_obj)[:12]}")
    elif callable(r65_obj):
        r65_lines.append(f"{r65_key}: callable")
    elif isinstance(r65_obj, (bi.list, bi.tuple)):
        r65_lines.append(f"{r65_key}: {bi.type(r65_obj).__name__}, length={bi.len(r65_obj)}")
print("\n".join(r65_lines)[:5800])

print("\n=== CONFIGURATION ===")
print(VOCAB["config"])
print("VOCAB keys:", bi.list(VOCAB))
print("\n=== STATE FILE METADATA ===")
for r65_path in sorted(Path("/work/fwd_3").glob("*.npz")):
    with np.load(r65_path, allow_pickle=False) as r65_npz:
        print(r65_path.name,
              [(k, r65_npz[k].shape, str(r65_npz[k].dtype)) for k in r65_npz.files])
print("\n=== STATE AND TOKEN SCALES ===")
for r65_key in ["sx", "hflow_all", "e_all", "u_all", "r40_context"]:
    r65_obj = globals()[r65_key]
    r65_arr = compact_array(r65_obj)
    print(r65_key, "rms", float(np.sqrt(np.mean(r65_arr*r65_arr))),
          "max", float(np.max(np.abs(r65_arr))))
print("Info sample:", r40_info[bi.int(single_start_idx[0])])

with Path("/work/notes.md").open("a") as r65_file:
    r65_file.write("\nPython run 65; forwards 6. Original model restored. "
                   "Authoritative states and action encodings retained; inspecting prelude "
                   "attention and later repair fits to identify an effective causal injection path.\n")
```
