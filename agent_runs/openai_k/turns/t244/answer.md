The rollout refutes this candidate: insulin remains far too weak, and norepinephrine is essentially unresponsive. Preserved factual fit is not enough. I’ll inspect the saved dose channels and calibration machinery to distinguish inactive channels from ineffective downstream routing; that distinction determines what to change next.

```python
import numpy as np
import torch
import inspect
from pathlib import Path

resume220_notes = Path("/work/notes.md").read_text()
print("=== EARLY EVIDENCE ===")
print(resume220_notes[:6000])

print("=== RELEVANT RETAINED OBJECTS ===")
resume220_terms = (
    "channel", "physical", "reserved", "baseline", "surrog",
    "replay", "probe", "pair", "source", "build", "candidate"
)
resume220_objects = [
    (k, v) for k, v in list(globals().items())
    if not k.startswith("resume") and any(t in k.lower() for t in resume220_terms)
]
for k, v in resume220_objects[:120]:
    shape = getattr(v, "shape", None)
    detail = tuple(shape) if shape is not None else ""
    if isinstance(v, dict):
        detail = "keys=" + str(list(v.keys())[:15])
    elif isinstance(v, str):
        detail = "characters=" + str(len(v))
    print(k, type(v).__name__, detail)

print("=== SAVED LATENT INJECTION ===")
resume220_zw = load("z_in.weight")
resume220_zo = load_original("z_in.weight")
print("z_in column-change norms:",
      torch.linalg.vector_norm(resume220_zw - resume220_zo, dim=0).tolist())
print("z_in last-two column norms:",
      torch.linalg.vector_norm(resume220_zw[:, -2:], dim=0).tolist())
print("D_best158 column norms:", np.linalg.norm(D_best158, axis=0))
print("candidate158 tensors:", list(candidate158.keys()))

print("=== REPLAY FUNCTION DEPENDENCIES ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and any(
        t in k.lower() for t in ["core", "head", "channel", "physical", "candidate", "routing", "replay"]
    ):
        print(k, "args=", v.__code__.co_varnames[:v.__code__.co_argcount],
              "dependencies=", v.__code__.co_names[:45])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 220: candidate158 rollout refuted\n"
        "- Forward 16: edited insulin x2/half -1.823/+1.007; NE x2/half -0.004/+0.009.\n"
        "- Factual delta remains +0.00002, but paired-state calibration did not repair causal rollout.\n"
        "- Inspecting saved physical-dose channels and downstream routing before another edit.\n"
        "- Python runs: 220; forward passes: 16.\n"
    )
```