The specification confirms that value predictions depend on both the recurrent state and a last-reading register. The cached reconstruction covers fewer readings than the factual file, so missing initial readings may explain the failure. I’ll check that coverage and recover the causal estimates before choosing another edit.

```python
import json
import numpy as np
import pandas as pd

print("=== FORWARD REQUEST METADATA ===")
for k in (3, 12, 13):
    p = Path(f"/work/fwd_{k}/summary.json")
    obj = json.loads(p.read_text())
    print(k, json.dumps(obj)[:1600])

print("\n=== STATE FILE ===")
sf = np.load("/work/fwd_3/states_original.npz")
print([(k, sf[k].shape) for k in sf.files])

print("\n=== HEAD METADATA ===")
print("columns:", list(head_meta.columns))
print(head_meta.head(8).to_string(index=False))
print("factual rows:", len(head_factual), "metadata rows:", len(head_meta))
print("cached head shapes:",
      {k: tuple(v.shape) for k, v in repair_base_head.items()
       if hasattr(v, "shape")})

print("\n=== INITIAL STATE AND HEAD TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("h0", "init", "last_", "value_bin", "transition", "type_emb")):
        print(name, SHAPES[name])

print("\n=== NOTEBOOK EARLY EVIDENCE ===")
text = Path("/work/notes.md").read_text()
print("notebook characters:", len(text))
print(text[:5200])

print("\n=== SAVED CAUSAL ESTIMATES ===")
for filename in ("dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv"):
    frame = pd.read_csv("/work/" + filename)
    print(filename, frame.shape, list(frame.columns))
    print(frame.head(5).to_string(index=False))
```
