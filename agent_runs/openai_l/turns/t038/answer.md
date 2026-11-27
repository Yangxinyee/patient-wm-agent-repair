The architecture offers two possible edit points: how actions enter the state, and how the value head reads that state. I’ll inspect their exact tensors and the retained data next. If the state already carries usable dose information, a readout edit may suffice; otherwise the action pathway will need repair.

```python
from pathlib import Path
import inspect
import json

print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if (name.startswith(("encode_action.", "core.z_post.", "core.z_prior.",
                         "core.z_in.", "core.z_head.", "attr_heads.value_bin."))
        or name in ("embed.numeric.weight", "embed.numeric.bias",
                    "last_numeric.weight", "last_numeric.bias")):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for key in VOCAB:
    if any(s in key.lower() for s in ("norm", "scale", "median")):
        text = repr(VOCAB[key])
        print(key, text[:1600])

print("\n=== DECODER FEATURE FUNCTION ===")
try:
    print(inspect.getsource(r32_features))
except Exception as ex:
    import dis
    dis.dis(r32_features)

print("\n=== RETAINED OBJECT NAMES ===")
print([k for k in sorted(globals()) if
       k.startswith(("r30", "r31", "r32", "st", "out22", "z22", "W22"))
       or k in ("train", "tuning", "df", "P", "A", "kernels19",
                "kernel_se19", "res17", "residuals", "states22")])

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
for p in Path("/work").glob("fwd_*/*states*.npz"):
    data = np.load(p)
    print(p.name, [(k, data[k].shape, str(data[k].dtype)) for k in data.files])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 34\n"
            "Recovered architecture: actions enter posterior and prelude, while "
            "a shared nonlinear value head reads the state plus code/register. "
            "Inspecting exact editable matrices and existing state coverage "
            "before choosing a residual-action edit. No edits.\n")
```