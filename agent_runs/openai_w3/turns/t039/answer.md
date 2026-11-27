The drug-start reconstructions are essentially exact, but the reading reconstructions still disagree with the saved states. That points to an observation-path issue, not a general weight-loading problem. I’ll inspect the architecture and existing helpers before editing; an explicit difference in observation processing would support this diagnosis.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSOR NAMES AND SHAPES ===")
for n in NAMES:
    print(n, tuple(SHAPES[n]))

print("\n=== LIVE HELPER FUNCTIONS ===")
for n, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            src = inspect.getsource(v)
        except (OSError, TypeError):
            src = "<source unavailable>"
        print(n, inspect.signature(v))
        if any(s in n.lower() for s in ("atom", "embed", "encode", "posterior", "bundle", "state", "recon")):
            print(src[:6500])

print("\n=== LIVE DATA OBJECTS ===")
for n, v in list(globals().items()):
    if n.startswith("_") or n in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(v, "shape") and not callable(getattr(v, "shape", None)):
        print(n, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(n, "dict", len(v), "keys:", list(v)[:12])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 34; forwards 3. Drug-start posterior reconstruction remains exact, but observations do not. Recovering architecture and live helper definitions to isolate the observation-path discrepancy before any weight edit.\n")
```