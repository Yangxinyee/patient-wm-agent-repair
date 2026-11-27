Removing both drug identities increased factual NLL by 0.00658, exceeding the 0.005 limit, so I will reject that edit. I’ll restore numeric gain 4 and review the architecture and earlier evidence for a more selective intervention. A useful edit must strengthen dose responses without crossing the factual-fit limit.

```python
import os, json, numpy as np, pandas as pd

undo()
undo()
print("RESTORED OPERATIONS:", ops())

print("\n=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== DATA README ===")
print(open("/data/README.md").read())

r111_notes = open("/work/notes.md").read()
print("\n=== EARLY NOTEBOOK ===")
print(r111_notes[:20000])

print("\n=== AVAILABLE HELPERS AND SAVED OBJECTS ===")
for r111_name, r111_obj in sorted(list(globals().items())):
    if r111_name.startswith("_") or r111_name.startswith("r111_"):
        continue
    if callable(r111_obj) and getattr(r111_obj, "__module__", "") == "__main__":
        print("FUNCTION", r111_name)
    elif isinstance(r111_obj, np.ndarray):
        print("ARRAY", r111_name, r111_obj.shape)
    elif isinstance(r111_obj, pd.DataFrame):
        print("FRAME", r111_name, r111_obj.shape, list(r111_obj.columns)[:16])
    elif isinstance(r111_obj, dict) and r111_name not in ("SHAPES", "VOCAB"):
        print("DICT", r111_name, list(r111_obj.keys())[:16])

print("\n=== INPUT AND CORE TENSORS ===")
for r111_name in NAMES:
    if any(r111_s in r111_name for r111_s in
           ("embed.", "core.", "query", "z_head", "attr_heads.value")):
        print(r111_name, SHAPES[r111_name])

with open("/work/notes.md", "a") as r111_f:
    r111_f.write(
        "\n## Restart / Python run 78\n"
        "- Forward 31 identity ablation increased factual NLL by +0.00658; rejected.\n"
        "- Removed both drug-code row ablations; restored numeric gain 4 alone.\n"
        "- Recovering architecture, early causal evidence, and saved helpers before another selective edit.\n"
        "- Forward passes used: 31; 19 remain.\n"
    )
```