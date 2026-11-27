The stronger connections stayed within the factual-fit limit on the tested training stays through scale 2.5, but larger scales became unstable. The copy error stopped the persistence test, so we still do not know whether these edits produce sustained effects. I’ll recover the exact interfaces and repair that test before changing weights. Sustained responses without amplification would support this approach; rapidly vanishing or exploding responses would refute it.

```python
import inspect
from pathlib import Path
import numpy as np
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nEARLY NOTEBOOK: identification and repair setup")
notes_recover162 = Path("/work/notes.md").read_text()
print(notes_recover162[:8500])

print("\nCURRENT EDITS", edited())
print("\nRETAINED FUNCTIONS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and any(s in name.lower() for s in
                            ["replay", "impulse", "encode_action", "make_context",
                             "transition", "reading", "nll"]):
        try:
            print(name, inspect.signature(obj))
        except (ValueError, TypeError):
            pass

print("\nRETAINED OBJECT SUMMARIES")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in
           ["repair_bank", "weights_channel", "weights_rejected",
            "impulse_results", "scale_results161", "lp_new136", "kernel137",
            "target154", "connection", "candidate160"]):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, {
                str(k): (tuple(v.shape) if hasattr(v, "shape")
                         else f"{type(v).__name__}[{len(v)}]" if hasattr(v, "__len__")
                         else type(v).__name__)
                for k, v in obj.items()
            })
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        else:
            print(name, type(obj).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 162\nRun-161 training scales through 2.5 remained within "
            "the factual tolerance, but scales >=3.5 showed recurrent amplification. "
            "Its Tensor.copy error prevented the subsequent impulse diagnostic. "
            "Recovering architecture and retained interfaces before further edits; "
            "no saved weights changed.\n")
```