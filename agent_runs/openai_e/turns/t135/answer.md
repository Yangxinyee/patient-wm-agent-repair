The new norepinephrine probes failed to recover dose innovations reliably and responded unevenly to doubling and halving, so I will not scale them up. I’ll now recover the record-derived targets and model’s dose pathway. A more selective edit would support this approach if it improves dose responses while staying within the factual-NLL limit.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
_notes116 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: EVIDENCE AND ACCEPTED EDITS ===")
print(_notes116[:27000])
print("\n=== KERNEL OBJECT INVENTORY ===")
for _name116, _obj116 in list(globals().items()):
    if _name116.startswith("_"):
        continue
    if callable(_obj116) and hasattr(_obj116, "__code__"):
        print("FUNCTION", _name116, _obj116.__code__.co_varnames[:_obj116.__code__.co_argcount])
    elif any(s in _name116.lower() for s in (
        "causal", "effect", "target", "direction", "baseline", "panel",
        "ne_", "ins_", "innov", "fit", "state", "fact95", "dose", "readout"
    )):
        if isinstance(_obj116, (np.ndarray, torch.Tensor)):
            print("ARRAY", _name116, tuple(_obj116.shape), str(_obj116.dtype))
        elif isinstance(_obj116, dict):
            print("DICT", _name116, "keys=", list(_obj116)[:20])
        elif isinstance(_obj116, (list, tuple)):
            print(type(_obj116).__name__, _name116, "length=", len(_obj116))
        elif isinstance(_obj116, (str, int, float, bool, np.number)):
            print("VALUE", _name116, str(_obj116)[:180])
print("\n=== SAVED EDIT ARCHIVES ===")
for _file116 in ("baseline77.npz", "ne_trial103.npz", "ne_reduced107.npz"):
    with np.load("/work/" + _file116) as _archive116:
        print(_file116, {k: _archive116[k].shape for k in _archive116.files})
with open("/work/notes.md", "a") as _f116:
    _f116.write("\n## Run 116: reject innovation-probe route; recover causal targets and dose pathway\n"
                "Run115 innovation probes have poor held-out innovation fits and uneven doubling/halving gains; none installed. "
                "Recovering original architecture, record-derived target estimates, accepted insulin edit and retained helpers. "
                "Reduced NE correction remains installed. Python runs: 116; forwards: 15.\n")
```