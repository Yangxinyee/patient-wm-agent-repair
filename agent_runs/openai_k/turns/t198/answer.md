The architecture confirms that doses reach predictions through a nonlinear state update, while recent readings also enter the value head directly. The diagnostic output was truncated, so it did not yet establish why the correction failed. I’ll retrieve only the relevant estimates and helper interfaces; this should distinguish a routing failure from a units or alignment error.

```python
import types
from pathlib import Path
import re

ss182 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("=== FIRST IDENTIFICATION NOTES ===")
print("\n".join(ss182[:7])[:3300])
print("\n=== SAVED CANDIDATE NOTES ===")
for s182 in ss182:
    title182 = s182.splitlines()[0] if s182.splitlines() else ""
    if re.search(r"\b(138|139|140|141|142)\b", title182):
        print(s182[:650])
print("\n=== RECENT OBJECTS ===")
for n182 in sorted(list(globals())):
    if re.search(r"(179|180)$", n182) and not n182.startswith("_"):
        x182 = globals()[n182]
        if isinstance(x182, dict):
            print(n182, {str(k): (tuple(v.shape) if hasattr(v, "shape") else
                                list(v)[:8] if isinstance(v, dict) else type(v).__name__)
                         for k, v in list(x182.items())[:18]})
        elif isinstance(x182, types.FunctionType):
            print(n182, "args", x182.__code__.co_varnames[:x182.__code__.co_argcount])
        else:
            print(n182, type(x182).__name__, getattr(x182, "shape", None),
                  "len=" + str(len(x182)) if isinstance(x182, (list, tuple)) else "")
print("\n=== EXACT-CORE / HEAD HELPERS ===")
for n182 in sorted(list(globals())):
    x182 = globals()[n182]
    if isinstance(x182, types.FunctionType) and any(t in n182.lower() for t in
            ["exact", "step", "head", "action", "query", "readout", "effect", "core"]):
        print(n182, x182.__code__.co_varnames[:x182.__code__.co_argcount])
print("\n=== IDENTIFICATION OBJECT NAMES ===")
print([n for n in sorted(globals()) if any(t in n.lower() for t in
       ["panel", "ident", "impulse", "persist", "causal", "theta", "effect"])][:75])
with open("/work/notes.md", "a") as f182:
    f182.write("\n## Run 182: bounded recovery\n"
               "- Architecture recovered; run 181 output was truncated before useful retained interfaces.\n"
               "- Retrieving bounded identification, candidate139, and recent object summaries.\n"
               "- Python runs: 182; forward passes: 10. Saved weights unchanged.\n")
```