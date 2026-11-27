The value head is a two-layer network, and the original factual NLL is 1.57499 on the first tuning sample. A head-only correction may preserve that fit if it responds to dose deviations rather than condition alone. I’ll inspect state alignment and normalization next; poor dose decodability would argue against this route.

```python
import json
import numpy as np
from pathlib import Path

print("=== FORWARD BASELINES ===")
for k in (2, 3):
    p = Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        print(k, p.read_text())

print("\n=== SAVED STATES ===")
print("fields:", S.files)
for key in S.files:
    a0 = S[key]
    print(key, a0.shape, a0.dtype)
print(state_meta.head(8).to_string(index=False))
for name in ("state_train_ids", "fwd_ids"):
    a0 = globals().get(name)
    if a0 is not None:
        print(name, "length", len(a0), "first ten", list(a0)[:10])

print("\n=== PANEL INDEX AND SAMPLE ===")
print("panel index:", panel.index.names)
print(panel.head(4).to_string())
print("feature index:", F.index.names)
print("state arrays preview:")
for key in S.files:
    a0 = S[key]
    if a0.ndim == 1:
        print(key, a0[:6])

print("\n=== HEAD, LATENT, AND INITIAL-STATE TENSORS ===")
for name in NAMES:
    if any(s in name for s in ("z_", "prior", "post", "head", "initial", "init", "last_", "type_emb",
                               "query_norm", "out_norm", "filter")):
        print(name, SHAPES[name])

print("\n=== NUMERIC VOCAB STRUCTURE ===")
numeric_vocab = VOCAB["numeric"]
print(type(numeric_vocab).__name__)
if isinstance(numeric_vocab, dict):
    print("keys:", list(numeric_vocab))
print(json.dumps(numeric_vocab)[:2300])
print("centres structure:", type(VOCAB["value_bin_centres"]).__name__)
print(str(VOCAB["value_bin_centres"])[:900])

print("\n=== LAST-NUMERIC COLUMN SIZES ===")
lnw = np.asarray(load_original("last_numeric.weight"))
print("column norms:", np.linalg.norm(lnw, axis=0).round(4))
print("column means:", lnw.mean(axis=0).round(4))
print("\n=== CURRENT EDITS ===")
print(edited())
```