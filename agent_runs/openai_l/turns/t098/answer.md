The posterior matches, but the recurrent states do not, so the latent probe cannot support an edit. I’ll recover the architecture and validated recurrence inputs first. Matching the exported states closely would validate further probes; persistent discrepancies would rule them out.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

snapshot90 = list(globals().items())
print("\n=== LOCAL FUNCTION INTERFACES ===")
for name, value in snapshot90:
    if callable(value) and getattr(value, "__module__", None) == "__main__":
        try:
            print(name, inspect.signature(value))
        except Exception:
            pass

print("\n=== RECONSTRUCTION VARIABLE INVENTORY ===")
terms90 = ("mem", "mask", "transition", "state", "prev", "eps",
           "decode", "posterior", "valid", "probe", "causal", "kernel")
for name, value in snapshot90:
    if any(term in name.lower() for term in terms90):
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(name, type(value).__name__, tuple(shape))
        elif isinstance(value, dict):
            print(name, "dict", list(value.keys())[:15])
        elif isinstance(value, (list, tuple)):
            print(name, type(value).__name__, "length", len(value))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 90 / recurrence recovery\n"
            "Run 89 posterior RMS=1.95e-5 but recurrent out RMS=.1299 "
            "and h RMS=11.94. Latent probe remains invalid; recovering "
            "architecture and validated memory inputs before any edit. "
            "No tensors edited.\n")
```