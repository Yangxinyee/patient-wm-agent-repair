The reconstructed head tracks the GPU closely, but its mean errors and higher NLL are still too large to trust for weight editing. No weights have changed. I’ll recover the architecture and retained analysis, then fix the reconstruction. Matching GPU predictions numerically would confirm that we are editing the right computation.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(_notes[:14000])

print("\n=== RETAINED VARIABLES ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.ismodule(_obj):
        continue
    if inspect.isfunction(_obj):
        try:
            _desc = str(inspect.signature(_obj))
        except Exception:
            _desc = ""
        print(_name, "function", _desc)
    elif hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, tuple(_obj.shape),
              str(getattr(_obj, "dtype", "")))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj), "keys:", list(_obj)[:16])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, "length", len(_obj),
              "sample:", str(_obj[:3])[:180])
    elif isinstance(_obj, (str, int, float, bool)):
        print(_name, repr(_obj)[:180])

print("\n=== TENSOR NAMES AND SHAPES ===")
for _name in NAMES:
    print(_name, SHAPES[_name])
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Restart recovery; Python 26, forward 4\n"
        "- No edits. CPU head approximation is not yet exact: best mean RMSE "
        "5.49 and NLL 1.718 versus GPU 1.635.\n"
        "- Recovering architecture and analysis before attempting a "
        "factual-preserving causal edit.\n"
    )
```