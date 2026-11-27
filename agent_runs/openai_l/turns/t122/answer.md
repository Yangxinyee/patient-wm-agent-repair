The architecture shows that a new dose signal must reach the recurrent state before it can influence later readings. The retained raw-input residual fits look promising, but the latent bottleneck remains unresolved. I’ll inspect the decoder helpers, causal targets, and token sensitivities to see whether that route can be amplified reliably; weak or inconsistent sensitivities would argue against it.

```python
import numpy as np
import torch
from pathlib import Path

print("=== RELEVANT TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_prior", "z_in", "z_head",
                            "core.prelude", "core.c_out", "core.b_in",
                            "attr_heads.value_bin", "encode_action")):
        print(n, SHAPES[n])

print("\n=== CALLABLE INTERFACES ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and not k.startswith("_"):
        c = v.__code__
        print(k, c.co_varnames[:c.co_argcount])

print("\n=== TOKEN SENSITIVITY ===")
g113 = np.asarray(Gtoken110)
print("Range/RMS:", float(g113.min()), float(g113.max()),
      float(np.sqrt(np.mean(g113**2))))
for code113 in np.unique(sensor_codes110):
    gg113 = g113[sensor_codes110 == code113]
    print("sensor", code113, "n", len(gg113),
          "mean-gradient norm", float(np.linalg.norm(gg113.mean(0))),
          "RMS", float(np.sqrt(np.mean(gg113**2))))
print("singular values:", np.linalg.svd(g113, compute_uv=False)[:10])

print("\n=== SELECTED CAUSAL NOTES ===")
text113 = Path("/work/notes.md").read_text()
sections113 = text113.split("\n## ")
matches113 = [
    s for s in sections113
    if any(t in s.lower() for t in
           ("corrected dml", "impulse", "accepted", "fwd_",
            "forward pass", "causal estimate", "drug effect"))
]
budget113 = 4600
for s in matches113:
    excerpt113 = s[:min(1000, budget113)]
    if not excerpt113:
        break
    print("## " + excerpt113)
    budget113 -= len(excerpt113)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 113 / latent-route feasibility\n"
            "Recovered architecture: dose residuals can enter z_post, but must propagate "
            "through z_in/prelude and recurrence. Inspecting token controllability and "
            "validated causal/decoder interfaces before an analytical edit. No edits.\n")
```