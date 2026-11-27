The posterior helper uses GELU, so the earlier linear and tanh checks did not reproduce it exactly. I’ll now inspect the GPU state fields and the private posterior rows. If the disagreement is only a state-layout mistake, those rows should match once indexed correctly.

```python
import numpy as np, torch
from pathlib import Path

print("=== GPU STATE ARRAYS ===")
gpu265 = {}
for model in ("original", "edited"):
    a = np.load(f"/work/fwd_25/states_{model}.npz")
    gpu265[model] = {k: a[k] for k in a.files}
    print(model, [(k, a[k].shape, str(a[k].dtype)) for k in a.files])
    for k in a.files:
        if a[k].ndim == 1:
            print(k, a[k][:12])
        elif a[k].ndim == 2 and a[k].shape[1] == 992:
            print(k, "first private coordinates:", a[k][:12, 960 + np.array([16, 12])])
print("=== PRIVATE POSTERIOR SUPPORT ===")
n266 = np.load("/work/neutral209.npz")
p0w266 = n266["z_post.0.weight"]
p0b266 = n266["z_post.0.bias"]
p2w266 = n266["z_post.2.weight"]
p2b266 = n266["z_post.2.bias"]
for j in (16, 12):
    nz = np.flatnonzero(np.abs(p2w266[j]) > 1e-8)
    print("output", j, "bias", p2b266[j], "hidden neurons", nz,
          "coefficients", p2w266[j, nz])
    for r in nz:
        print(" neuron", int(r), "bias", p0b266[r],
              "block norms", [float(np.linalg.norm(p0w266[r, t:t+192]))
                              for t in (0, 192, 384)],
              "largest coefficients",
              [(int(c), float(p0w266[r, c]))
               for c in np.argsort(np.abs(p0w266[r]))[-8:]])
print("=== EXACT GELU LOCAL CHECK ===")
xx266 = torch.as_tensor(input196, dtype=torch.float32)
zz266 = torch.nn.functional.linear(
    torch.nn.functional.gelu(torch.nn.functional.linear(
        xx266, torch.from_numpy(p0w266), torch.from_numpy(p0b266))),
    torch.from_numpy(p2w266), torch.from_numpy(p2b266))
inactive266 = torch.linalg.vector_norm(xx266[:, 384:], dim=1) == 0
print("inactive count", int(inactive266.sum()),
      "private RMS", zz266[inactive266][:, [16, 12]].square().mean(0).sqrt().tolist(),
      "private min", zz266[inactive266][:, [16, 12]].amin(0).tolist(),
      "private max", zz266[inactive266][:, [16, 12]].amax(0).tolist())
print("=== RETAINED STATE VARIABLE NAMES ===")
print([k for k in globals()
       if any(t in k.lower() for t in ("gpu", "state", "private", "inactive"))
       and not k.startswith("_")][-65:])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 266 — inspect exact GELU and GPU state indexing\n"
            "- No weights changed; neutral209 remains installed.\n"
            "- Comparing the actual private posterior support and GPU array layout, not surrogate activations.\n")
```