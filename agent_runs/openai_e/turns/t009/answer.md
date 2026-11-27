The original model’s factual NLL is 1.63461, so the allowed ceiling on this sample is 1.63961. The record-based drug-effect estimates remain provisional. Next I’ll recover the architecture and retained analysis variables to identify where a weight-only correction could change dose sensitivity without disrupting factual predictions. A useful correction must improve the dose rollouts while staying below that NLL ceiling.

```python
from pathlib import Path
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, types.ModuleType) or callable(v):
        continue
    desc = type(v).__name__
    if hasattr(v, "shape"):
        desc += f" shape={v.shape}"
    elif isinstance(v, (list, tuple, dict, set)):
        desc += f" len={len(v)}"
    elif isinstance(v, (str, int, float, bool)):
        desc += f" value={str(v)[:160]}"
    print(k, desc)

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
for k, v in VOCAB.items():
    if isinstance(v, dict):
        print(k, {kk: (vv if not isinstance(vv, (list, dict)) else
                       f"{type(vv).__name__}[{len(vv)}]") for kk, vv in v.items()})
    else:
        print(k, str(v)[:200])

print("\n=== FULL NOTEBOOK ===")
print(Path("/work/notes.md").read_text())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 7: restart and factual baseline\n"
            "Original factual value NLL on the fixed 100-stay tuning sample: 1.63461 "
            "(4751 readings); allowed sample ceiling 1.63961. "
            "Recovering architecture and retained analysis state before choosing an edit.\n"
            "Python runs: 7. Forward requests: 2.\n")
```