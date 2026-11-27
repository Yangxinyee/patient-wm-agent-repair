The token changes barely moved the drug-response slopes, even at large gains, so that route does not yet look useful. I’ll recover the architecture, record-derived effect estimates, and earlier validation results before changing more weights. Those should show whether the remaining error is in dose transport or in how the model turns dose information into readings.

```python
from pathlib import Path
import json, re, numpy as np

arch = Path("/model/ARCHITECTURE.md").read_text()
notes = Path("/work/notes.md").read_text()
print("ARCHITECTURE\n", arch)
print("\nDATA SCHEMA\n", Path("/data/README.md").read_text())
print("\nNOTEBOOK BEGINNING\n", notes[:12500])
print("\nRELEVANT NOTEBOOK LINES")
hits = [line for line in notes.splitlines()
        if re.search(r"true|causal|instrument|random|effect|coefficient|slope|NLL|rollout", line, re.I)]
for line in hits[-100:]:
    print(line[:500])
print("\nCURRENT EDITS", edited())
print("\nRETAINED VARIABLES")
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if callable(value) and getattr(value, "__module__", "") == "__main__":
        print("FUNCTION", key)
    elif isinstance(value, np.ndarray) and value.size > 16:
        print("ARRAY", key, value.shape)
    elif isinstance(value, dict) and re.search(r"fit|effect|candidate|route|replay|stat|target|base", key, re.I):
        print("DICT", key, list(value)[:25])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 253 — recover causal targets and architecture\n"
            "- Direct-token gains saturated without a useful response correction.\n"
            "- Reading architecture, schema, retained helpers, and previous record evidence before further edits.\n"
            "- No weights changed.\n")
```