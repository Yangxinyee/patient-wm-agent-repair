The architecture confirms that doses enter the recurrent state, while reading predictions also use explicit last-reading registers. That gives us potential targeted edits, but the output was truncated before the tensor details. I’ll recover those details and the earlier causal estimates; they should clarify whether the reconstruction error is a feature-layout mistake or a deeper mismatch.

```python
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK SECTION INDEX ===")
print("\n".join(line for line in notes.splitlines() if line.startswith("## ")))
print("\n=== EARLIER EVIDENCE, CONTINUED ===")
print(notes[3000:6900])

print("\n=== RELEVANT TENSORS ===")
prefixes = (
    "attr_heads.value", "last_", "type_emb", "query_norm",
    "embed_transition.value", "core.gap", "core.z_prior",
    "core.z_head", "core.out_norm"
)
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== CONFIG AND CODES ===")
print(VOCAB.get("config"))
print(VOCAB.get("codes"))

print("\n=== KERNEL FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if inspect.isfunction(v) and not k.startswith("_")])

print("\n=== KERNEL DATA OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"notes"}:
        continue
    if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict) and k not in {"SHAPES", "VOCAB"}:
        print(k, "dict", len(v), "keys:", list(v)[:8])
```