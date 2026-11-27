The affine dose decoders failed on held-out training stays and recovered only a fraction of dose changes. I will not use them for an edit. Next I’ll inspect the earlier action pathway and causal estimates; a usable pathway must respond reliably to dose changes without disturbing factual predictions.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND BASELINE ===")
audit_notes = Path("/work/notes.md").read_text()
print(audit_notes[:13000])

print("\n=== RETAINED FUNCTIONS ===")
audit_globals = list(globals().items())
for audit_name, audit_obj in audit_globals:
    if inspect.isfunction(audit_obj) and getattr(audit_obj, "__module__", "") == "__main__":
        try:
            print(audit_name, inspect.signature(audit_obj))
        except Exception:
            print(audit_name)

print("\n=== RELEVANT RETAINED ARRAYS / TABLES ===")
audit_words = ("post", "hidden", "innov", "panel", "rate", "feature", "dir",
               "level", "ctx", "act", "latent", "target", "decod", "impulse",
               "lag", "bundle", "drug")
audit_count = 0
for audit_name, audit_obj in audit_globals:
    if not audit_name.startswith("audit_") and any(w in audit_name.lower() for w in audit_words):
        if hasattr(audit_obj, "shape"):
            print(audit_name, type(audit_obj).__name__, audit_obj.shape)
            audit_count += 1
        elif isinstance(audit_obj, dict):
            print(audit_name, "dict", list(audit_obj.keys())[:12])
            audit_count += 1
        elif isinstance(audit_obj, (list, tuple)):
            print(audit_name, type(audit_obj).__name__, len(audit_obj))
            audit_count += 1
        if audit_count >= 100:
            print("(inventory capped)")
            break

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 120 — reject weak posterior dose decoders\n"
            "- Held-stay innovation R2 was negative; counterfactual dose slopes were far below one.\n"
            "- Do not submit an edit based on these decoders. Recovering earlier action features and causal evidence.\n"
            "- No weights changed.\n")
print("\nCurrent edits:", edited())
```