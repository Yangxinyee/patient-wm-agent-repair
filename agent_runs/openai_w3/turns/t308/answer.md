The repair passes through several nonlinear state updates before reaching the value head. I’ll test a weaker **whole attention-output and compensation edit**, rather than the isolated carrier gains tried earlier. If excessive state displacement causes the drift, this should improve rollout symmetry; otherwise I’ll restore the saved edit.

```python
import torch, numpy as np, re
from pathlib import Path

checkpoint273 = {name: load(name).clone() for name in edited()}
scaled_names273 = [
    "core.prelude.0.attn.out.weight",
    "core.prelude.0.ffn.2.weight",
    "core.prelude.0.ffn.2.bias",
]
alpha273 = 0.5
for name in scaled_names273:
    original = load_original(name)
    candidate = original + alpha273 * (checkpoint273[name] - original)
    assert tuple(candidate.shape) == tuple(SHAPES[name])
    assert torch.isfinite(candidate).all()
    save_edit(name, candidate)
    print(name, "original_norm", float(original.norm()),
          "old_delta_norm", float((checkpoint273[name] - original).norm()),
          "new_delta_norm", float((candidate - original).norm()))

print("edited:", edited())
print("=== COMPENSATION CACHE INVENTORY ===")
for name, obj in list(globals().items()):
    if re.search(r"tokens|(^[CEUH]154$)|post154|zin154|zb154|orig.*raw|raw.*orig", name, re.I):
        if isinstance(obj, dict):
            print(name, "keys", list(obj)[:12])
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, "length", len(obj))

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 273; forwards 22. Installed reversible whole-output-path scaling experiment: attention out and FFN output weight/bias deltas multiplied by .5; latent encoder and attention q/kv unchanged. This differs from prior isolated carrier payload scaling. checkpoint273 retains the factually validated gamma=.1 candidate. Next factual safeguard, then rollout if safe.\n")
```
