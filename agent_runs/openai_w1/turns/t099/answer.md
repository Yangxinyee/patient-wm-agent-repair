The last result shows that both drug encodings now match the exported posteriors to numerical precision. This removes the reconstruction error that blocked reliable calibration. No weights are edited.

Next I’ll recover the architecture and retained candidates. A candidate must reproduce dose-dependent states on held-out records before we test whether it improves rollouts without damaging factual fit.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== LOCAL FUNCTION INDEX ===")
r90_functions = []
for r90_name, r90_obj in list(globals().items()):
    if inspect.isfunction(r90_obj) and getattr(r90_obj, "__module__", "") == "__main__":
        try:
            r90_sig = str(inspect.signature(r90_obj))
        except Exception:
            r90_sig = "(signature unavailable)"
        r90_functions.append((r90_name, r90_sig))
for r90_name, r90_sig in sorted(r90_functions):
    print(r90_name, r90_sig)

print("\n=== TRACE CANDIDATE STRUCTURE ===")
for r90_name in ["rr51_trace_candidates", "rr51_assignment"]:
    r90_obj = globals().get(r90_name)
    if isinstance(r90_obj, dict):
        for r90_key, r90_value in r90_obj.items():
            print(r90_name, repr(r90_key), type(r90_value).__name__)
            if isinstance(r90_value, dict):
                for r90_k2, r90_v2 in r90_value.items():
                    print(" ", repr(r90_k2), type(r90_v2).__name__,
                          getattr(r90_v2, "shape", ""),
                          repr(r90_v2) if np.isscalar(r90_v2) else "")
            elif isinstance(r90_value, (tuple, list)):
                print(" ", [(type(v).__name__, getattr(v, "shape", "")) for v in r90_value])
    else:
        print(r90_name, type(r90_obj).__name__)

with open("/work/notes.md", "a") as r90_f:
    r90_f.write(
        "\n## Run 90\n"
        "Drug and measurement encoder reconstruction is now exact after correcting categorical units. "
        "Recovering architecture and retained trace candidates before any weight edit. "
        "No tensors edited. Counts: 90 Python runs, 6 forward passes.\n"
    )
```
