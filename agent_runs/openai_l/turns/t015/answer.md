The state export succeeded, but it does not establish drug effects. The insulin threshold estimates remain stronger evidence than the conflicting state-space fits. I’ll recover the latest record analysis and model architecture before choosing an edit; agreement across independent estimators and sensors would strengthen the causal case.

```python
import os, json, numpy as np
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== PERSISTENT VARIABLE INVENTORY ===")
for k, v in list(globals().items()):
    if not k.startswith("_") and k not in {"NAMES", "SHAPES", "VOCAB"}:
        typ = type(v).__name__
        if typ not in {"module", "function", "builtin_function_or_method", "type"}:
            print(k, typ, getattr(v, "shape", ""),
                  ("keys=" + str(list(v.keys())[:30])) if isinstance(v, dict) else "")
print("\n=== VOCAB KEYS ===", list(VOCAB.keys()))
print("\n=== STATE ARCHIVE ===")
s3 = np.load("/work/fwd_3/states_original.npz")
for k in s3.files:
    print(k, s3[k].shape, s3[k].dtype)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 12 / second restart\nRecovered architecture and persistent-variable inventory after original state export. No edits; insulin threshold evidence remains provisional, and neither conflicting SSM is accepted as causal truth. Next: inspect flexible residualization results and state/head pathways.\n")
```
