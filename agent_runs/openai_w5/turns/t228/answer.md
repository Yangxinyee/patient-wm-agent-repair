The earlier record analysis found a negative insulin effect on glucose and a positive norepinephrine effect on MAP, with uncertainty in their longer-term persistence. The current rollout is much less symmetric than those estimates suggest. I’ll preserve this candidate and identify its private channels and head edits before making further changes.

```python
from pathlib import Path
import numpy as np, torch
from safetensors.torch import save_file

W = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in NAMES}
WO = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES}
save_file({n: W[n].contiguous() for n in edited()},
          "/work/checkpoint_run202.safetensors")

print("=== NOTEBOOK CONTINUATION ===")
print(Path("/work/notes.md").read_text()[6500:12500])

print("\n=== PRIVATE-CHANNEL NORMALIZATION AND DECAY ===")
for n in ["core.flow.a", "core.delta_in", "core.out_norm.weight",
          "core.out_norm.bias", "query_norm.weight", "query_norm.bias",
          "core.prelude_norm.weight", "core.prelude_norm.bias"]:
    d = (W[n] - WO[n]).abs()
    idx = torch.where(d > 1e-5)[0]
    print(n, "changed", idx.tolist(),
          "values", W[n][idx].tolist(),
          "original", WO[n][idx].tolist())

print("\n=== VALUE-HEAD ROW CHANGES ===")
for n in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
          "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
    d = W[n] - WO[n]
    if d.ndim == 2:
        norms = d.norm(dim=1)
        idx = norms.argsort(descending=True)[:16]
        print(n, "largest row changes",
              [(int(i), round(float(norms[i]), 5)) for i in idx])
    else:
        idx = torch.where(d.abs() > 1e-4)[0]
        print(n, [(int(i), round(float(W[n][i]), 5),
                   round(float(WO[n][i]), 5)) for i in idx[:24]])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 202: clean recovery after timeout\n"
            "- Kernel variables were lost in run 199; persisted edits survived.\n"
            "- Saved all current edits in /work/checkpoint_run202.safetensors.\n"
            "- Rebuilt weight caches directly from the current and original tensors; no weights changed.\n")
```