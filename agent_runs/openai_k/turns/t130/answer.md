The last test did not resolve the mismatch between local predictions and the forward tool, so local factual-loss estimates remain unreliable. I’ll recover the architecture and the record-derived drug effects, then validate candidate edits with the forward tool. An edit is supported only if it produces the identified effects while increasing factual value NLL by no more than 0.005 nats.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())
notes121 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND BASELINES ===")
print(notes121[:16000])
print("\n=== RETAINED OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB", "notes121"}:
        continue
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print("FUNCTION", k, inspect.signature(v))
        except Exception:
            pass
    elif isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 2:
        print("ARRAY", k, tuple(v.shape))
    elif isinstance(v, dict) and any(s in k.lower() for s in ["effect", "target", "candidate", "fit", "result", "baseline"]):
        print("DICT", k, list(v.keys())[:20])
print("\n=== TENSORS ===")
for n in NAMES:
    print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 121: external-validation reset\n"
            "- Local readout conventions remain inconsistent with official predictions; local factual scores are not trusted.\n"
            "- Recovering architecture, record identification, and repair interfaces before forward validation.\n"
            "- No submitted edits. Python runs: 121; forward passes: 4.\n")
```
