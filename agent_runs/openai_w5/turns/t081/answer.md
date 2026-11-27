The fitted directions matched tiny perturbations, but finite-dose responses were unstable and affected the wrong channel. I conclude that those directions are not ready to use. I’ll reread the architecture and inspect the saved analysis objects to find a simpler candidate. Consistent finite-dose responses and a passing GPU factual test would support it; instability or excess factual loss would refute it.

```python
from pathlib import Path
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== PERSISTENT ANALYSIS OBJECTS ===")
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(value, types.FunctionType):
        code = value.__code__
        print("FUNCTION", key, "args:", code.co_varnames[:code.co_argcount])
    elif hasattr(value, "shape") and not callable(getattr(value, "shape")):
        print("ARRAY", key, tuple(value.shape))
    elif isinstance(value, dict) and key not in ("SHAPES", "VOCAB", "__builtins__"):
        print("DICT", key, "keys:", list(value)[:18])
    elif isinstance(value, list) and key != "NAMES":
        print("LIST", key, "len:", len(value))

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after run 68\n"
            "- No edits retained. Infinitesimal latent-response fits did not yield stable finite-innovation responses.\n"
            "- Re-reading architecture and inspecting persistent objects before selecting a finite-dose, GPU-validated candidate.\n"
            "- Small-perturbation agreement alone is not evidence of causal rollout accuracy.\n")
```