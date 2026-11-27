Held-out NLL rose by 0.00821 nats, so this candidate fails the factual-fit constraint. I’ll recover the previous edit and replay interfaces to separate channel-construction damage from response strength. If the insulin-only baseline still passes, the NE extension needs revision.

```python
from pathlib import Path
import re, inspect, numpy as np, pandas as pd

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("DATA README\n", Path("/data/README.md").read_text())
notes196 = Path("/work/notes.md").read_text()
print("NOTEBOOK LENGTH", len(notes196))
relevant196 = [
    line for line in notes196.splitlines()
    if re.search(r"rollout|forward|identif|readout|causal|NLL|kernel", line, re.I)
]
print("RECENT EVIDENCE\n", "\n".join(relevant196[-45:]))
print("RETAINED INTERFACES")
for name196, obj196 in sorted(list(globals().items())):
    if name196.startswith("_"):
        continue
    if callable(obj196) and re.search(r"replay|dual|channel|capture|response|eval|build|fit.*ne|edit", name196, re.I):
        try:
            print(name196, inspect.signature(obj196))
        except Exception:
            print(name196, type(obj196).__name__)
    elif re.search(r"backup_before195|results194|193|194|195|record.*kernel|readout.*est", name196, re.I):
        if isinstance(obj196, dict):
            print(name196, "dict keys:", list(obj196)[:25])
        elif hasattr(obj196, "shape"):
            print(name196, type(obj196).__name__, tuple(obj196.shape))
        elif isinstance(obj196, (list, tuple)):
            print(name196, type(obj196).__name__, "length", len(obj196))
        elif not callable(obj196):
            print(name196, type(obj196).__name__, str(obj196)[:350])
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 196\nGPU factual validation on 98 tuning stays failed: original 1.59352, edited 1.60173, delta +.00821. Recovering architecture, evidence and retained candidate/replay interfaces to isolate the NE extension's damage before revising it. No weights changed.\n")
```