The architecture confirms that dose information enters the recurrent state, while the reading head also uses a separate last-reading register. That offers a possible way to isolate dose sensitivity without disturbing ordinary predictions. I’ll recover the causal estimates and repair helpers next; stable treatment estimates and a separable dose signal would support that approach.

```python
import inspect, json, re

print("=== NOTEBOOK IDENTIFICATION AND DESIGN SUMMARY ===")
for line in notes_text.splitlines():
    if line.startswith("## ") or (
        line.startswith("- ") and
        any(s in line.lower() for s in (
            "threshold", "discontinu", "instrument", "causal", "persistence",
            "rho", "identif", "target", "register", "feature", "decoder",
            "confound", "negative control", "balance", "placebo"
        ))
    ):
        print(line)

print("\n=== HELPER SIGNATURES ===")
for n in sorted(helper_names):
    try:
        print(n, inspect.signature(globals()[n]))
    except Exception:
        print(n)

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for k in VOCAB:
    if any(s in k.lower() for s in ("norm", "code", "unit", "config", "categor")):
        txt = repr(VOCAB[k])
        print(k, txt[:1800])

print("\n=== RELEVANT GLOBAL NAMES ===")
relevant_names = sorted([
    n for n in list(globals())
    if not n.startswith("_") and any(s in n.lower() for s in (
        "rm_", "spec", "poster", "graft", "robust", "panel", "ridge",
        "beta", "rho", "gate", "decod", "state", "head", "check", "rd_"
    ))
])
print(" ".join(relevant_names))
```
