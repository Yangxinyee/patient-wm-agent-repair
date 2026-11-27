The architecture has a separate action pathway, which could allow a dose-sensitive edit without changing the reading inputs. Earlier analyses found roughly −2 glucose units per insulin unit and +3 arterial MAP units per norepinephrine rate unit, but the recovered output was cut off before the dynamics checks. I’ll recover those checks and the reconstruction code before deciding how to edit.

```python
import numpy as np
import torch
import inspect
from pathlib import Path

text = Path("/work/notes.md").read_text()
start = text.find("## Run 6:")
end = text.find("## Run 15:")
print("=== CAUSAL DYNAMICS AND RECONSTRUCTION NOTES ===")
print(text[start:end] if start >= 0 else text[:6000])

print("\n=== FUNCTIONS DEFINED IN THE KERNEL ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        print(name, str(inspect.signature(obj)),
              "uses:", obj.__code__.co_names[:20])

print("\n=== IMPORTANT PERSISTENT OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or callable(obj) or name in ("text", "notes_text"):
        continue
    typ = type(obj).__name__
    if typ in ("ndarray", "Tensor", "DataFrame"):
        print(name, typ, tuple(obj.shape))
    elif typ == "dict" and name not in ("SHAPES", "VOCAB", "__builtins__"):
        print(name, "dict", len(obj), "keys:", list(obj)[:10])

print("\n=== CORE TENSOR LAYOUT ===")
for name in NAMES:
    if name.startswith(("core.", "z_", "gap.", "out_norm.")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 29: targeted restart recovery\n"
            "- Architecture recovered; separate action encoder and posterior latent pathway are candidate edit boundaries.\n"
            "- Recovering dynamic identification results and reconstruction objects before selecting an edit.\n"
            "- Python runs: 29; forward passes: 4; no weights edited.\n")
```