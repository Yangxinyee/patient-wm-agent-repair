The architecture shows that doses enter a separate action encoder, then affect the shared recurrent state and reading head. A targeted edit may be possible, but the previous output was truncated. I’ll recover the remaining head details and earlier causal evidence; these will guide an edit that changes dose sensitivity without disrupting factual predictions.

```python
from pathlib import Path
import re

print("=== REMAINING RELEVANT TENSORS ===")
for name in NAMES:
    if name.startswith(("last_", "embed_transition", "query_", "type_", "z_", "gap.",
                        "embed.numeric", "embed.kq", "encode_action", "filter_")):
        print(name, SHAPES[name])

print("\n=== VOCAB KEYS ===")
print(list(VOCAB))
for key in ("config", "codes", "normalisation", "normalization",
            "value_bin_centres", "rate_bin_centres"):
    if key in VOCAB:
        print(key, repr(VOCAB[key])[:2300])

print("\n=== HELPER SIGNATURES ===")
for name in ("cpu_head_parts", "cpu_head_probs", "repair_head_features"):
    fn = globals().get(name)
    if callable(fn):
        print(name, "args", fn.__code__.co_varnames[:fn.__code__.co_argcount],
              "defaults", fn.__defaults__, "references", fn.__code__.co_names)

print("\n=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:3800])

print("\n=== RETAINED HEAD VARIABLES ===")
for name in sorted(list(globals())):
    if name.startswith(("head_", "reference_", "reg_", "state_files")):
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", ""),
              list(obj.keys())[:8] if isinstance(obj, dict) else "")
```
