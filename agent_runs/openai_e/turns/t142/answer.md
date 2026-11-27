```python
from pathlib import Path
import inspect, types, dis, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT HELPERS ===")
helper123 = []
for name, obj in list(globals().items()):
    if isinstance(obj, types.FunctionType):
        refs = set(obj.__code__.co_names)
        if ({"src45", "centres45"} <= refs
                or name.endswith(("120", "121", "122"))
                or any(s in name.lower() for s in ("coda", "readout", "decoder"))):
            helper123.append((name, obj))
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "?"
            print(name, sig, "globals:", sorted(refs))

print("\n=== TRUSTED DECODER, COMPLETE ===")
for name, obj in helper123:
    if {"src45", "centres45"} <= set(obj.__code__.co_names):
        print("FUNCTION:", name)
        dis.dis(obj)

print("\n=== RETAINED READOUT INPUTS ===")
for name, obj in list(globals().items()):
    if name.endswith(("120", "121", "122", "45")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape), str(obj.dtype))
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:24])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 123: recover exact decoder semantics\n"
            "Register offsets alone did not fix the decoder. Reading architecture and complete trusted helper "
            "before using any local output constraints. No weights changed; reduced NE edit remains installed. "
            "Python runs: 123; forwards: 15.\n")
```