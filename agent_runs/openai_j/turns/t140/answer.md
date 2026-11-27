The architecture allows a selective value-head edit, avoiding direct changes to the other output heads. The notebook also shows that the drug-effect estimates remain uncertain, especially insulin’s persistence. The output was truncated before the current correction variables, so I’ll retrieve those and the identification checks before choosing an edit.

```python
import re
from pathlib import Path

snapshot126 = list(globals().items())
print("=== FUNCTION NAMES ===")
print([k for k, v in snapshot126
       if callable(v) and hasattr(v, "__code__") and not k.startswith("_")])

print("\n=== RECENT VARIABLES ===")
for k, v in snapshot126[-65:]:
    if k.startswith("_") or k in {"notes_text", "obj", "snapshot126"}:
        continue
    if hasattr(v, "shape"):
        detail = str(tuple(v.shape))
    elif isinstance(v, dict):
        detail = f"{len(v)} keys: {list(v)[:10]}"
    elif isinstance(v, (list, tuple)):
        detail = f"length {len(v)}: {repr(v[:6])[:120]}"
    elif isinstance(v, (str, int, float, bool)):
        detail = repr(v)[:150]
    else:
        detail = ""
    print(k, type(v).__name__, detail)

print("\n=== IDENTIFICATION CHECKS ===")
nt126 = Path("/work/notes.md").read_text()
start126 = nt126.find("## Causal identification checks")
print(nt126[start126:start126 + 4400])

print("\n=== CORRECTION-RELATED VARIABLE NAMES ===")
print([k for k, v in snapshot126 if re.search(
    r"coef|compress|drop|unit|insulin|tilt|candidate|contrast|correct|tau|decoder", k, re.I
) and not k.startswith("_")])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart audit, runs 126–127\n"
            "- Re-read architecture; no active edits. Selective value-head edits avoid direct changes to other heads.\n"
            "- Run 125 insulin-only correction had favourable local CV loss, and two-unit compression was nearly lossless.\n"
            "- These remain local approximations; require weight-equivalence and GPU factual/rollout validation.\n")
```
