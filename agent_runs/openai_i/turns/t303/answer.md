The empty-set encoder produces a nonzero vector, and using it reproduces the unwanted untreated signal. I’ll construct a gate correction that suppresses this vector while leaving drug-containing bundles unchanged. Small errors on those bundles would support trying the corrected response route again.

```python
import numpy as np, torch, inspect
from pathlib import Path

ua268 = xx266[:, 384:].numpy()
active268 = ~inactive266.numpy()
xa268 = np.column_stack([ua268[active268], np.ones(active268.sum())]).astype(np.float64)
ve268 = np.r_[empty267[False].numpy().astype(np.float64), 1.0]
_, ss268, vt268 = np.linalg.svd(xa268, full_matrices=False)
print("=== ACTION-SUBSPACE SINGULAR VALUES ===")
print([(k, float(ss268[k])) for k in (0, 15, 31, 63, 95, 127, 159, 179, 189, 192)
       if k < len(ss268)])
gate_candidates268 = {}
for rank in (16, 32, 48, 64, 96, 128, 160, 180, 190, 192):
    if rank >= len(vt268):
        continue
    residual = ve268 - vt268[:rank].T @ (vt268[:rank] @ ve268)
    if residual @ ve268 < 1e-12:
        continue
    delta = -40.0 * residual / (residual @ ve268)
    error = xa268 @ delta
    gate_candidates268[rank] = delta
    print("rank", rank, "correction norm", float(np.linalg.norm(delta)),
          "active RMS", float(np.sqrt(np.mean(error**2))),
          "active max", float(np.max(np.abs(error))),
          "empty shift", float(ve268 @ delta))
print("=== CURRENT GATE RANGES ===")
pre268 = xx266 @ torch.from_numpy(p0w266).T + torch.from_numpy(p0b266)
hid268 = np.array([1, 105, 74, 166])
for r in hid268:
    print("neuron", int(r), "active quantiles",
          np.quantile(pre268[active268, r].numpy(), [0, .05, .5, .95, 1]))
print("other posterior outputs using gate neurons",
      [(int(r), np.flatnonzero(np.abs(p2w266[:, r]) > 1e-8).tolist())
       for r in hid268])
print("=== EARLIER NEUTRAL/RESPONSE NOTES ===")
notes268 = Path("/work/notes.md").read_text()
for tag in ("## Python 203", "## Python 209", "## Python 216", "## Python 233"):
    start = notes268.find(tag)
    if start >= 0:
        print(notes268[start:start+1800])
print("=== AVAILABLE RESPONSE HELPERS ===")
print([(k, str(inspect.signature(v))) for k, v in list(globals().items())
       if callable(v) and any(t in k.lower() for t in
                              ("encode_rows", "install", "response", "replay", "direction"))])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 268 — empty-action gate correction design\n"
            "- The encoded empty action set reproduces the GPU untreated private signal.\n"
            "- Designing a common gate shift that annihilates this vector while preserving observed action bundles.\n"
            "- No new weights installed; neutral209 remains active.\n")
```