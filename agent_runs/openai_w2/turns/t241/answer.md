```python
from pathlib import Path
import numpy as np, torch, inspect, hashlib

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK\n", notes_text[:10000])
evidence_lines = [
    line for line in notes_text.splitlines()
    if any(s in line.lower() for s in
           ["target", "instrument", "threshold", "persistence", "causal",
            "effect estimate", "record-based", "true effect"])
]
print("\nRECORD-EVIDENCE NOTES\n", "\n".join(evidence_lines[-90:]))

snapshot207 = dict(globals())
print("\nKERNEL VARIABLE NAMES\n",
      " ".join(sorted(k for k in snapshot207 if not k.startswith("_"))))
print("\nRELEVANT OBJECTS")
for k, v in sorted(snapshot207.items()):
    if any(s in k.lower() for s in
           ["206", "205", "204", "203", "constr", "pool",
            "current", "validated", "decoder", "probe"]):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys:", list(v)[:12])
        elif callable(v):
            try:
                print(k, "function", str(inspect.signature(v)))
            except Exception:
                print(k, "callable")
        else:
            print(k, type(v).__name__, repr(v)[:240])

print("\nCURRENT EDITS", edited())
print("core.b_in.weight SHA256",
      hashlib.sha256(load("core.b_in.weight").numpy().tobytes()).hexdigest())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run207: restart recovery\n"
            "No weight changes. Recovering architecture, record-effect evidence, "
            "and constrained-decoder definitions before a direct factual test. "
            "Run206 residual reductions remain proxy evidence only.\n")
```