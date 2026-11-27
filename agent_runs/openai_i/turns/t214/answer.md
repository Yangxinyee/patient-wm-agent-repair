The records show the key confounding pattern: unadjusted dose associations have the opposite signs from history-adjusted estimates. The adjusted estimates consistently support lower glucose after insulin and higher arterial MAP after norepinephrine, though they remain observational estimates. I’ll inspect the retained state-update helpers to find where a dose-minus-policy correction could enter; it must survive into later predictions to be useful.

```python
import inspect, re
print("=== PRIOR PATHWAY TESTS ===")
sections190 = re.split(r"(?=\n## )", notes189)
for s190 in sections190:
    if any(x in s190.lower() for x in ["norm scaled", "scale action", "scaling", "python 31", "python 32", "python 33", "python 34", "python 35"]):
        print(s190[:1500])

print("\n=== COMPUTATION HELPERS ===")
for n190, o190 in list(globals().items()):
    if callable(o190) and getattr(o190, "__module__", "") == "__main__":
        if any(x in n190.lower() for x in ["replay", "step", "core", "readout", "encode", "feature", "register", "head", "context"]):
            try:
                print(n190, str(inspect.signature(o190))[:140],
                      "refs", str(o190.__code__.co_names)[:260])
            except Exception:
                pass

print("\n=== RELEVANT WEIGHTS ===")
for n190 in NAMES:
    if any(x in n190 for x in ["core.prelude", "core.b_in", "core.c_out", "encode_action", "core.coda"]):
        print(n190, SHAPES[n190])

print("\n=== RETAINED ARRAYS, RUNS 176–186 ===")
for n190, o190 in list(globals().items()):
    if re.search(r"(176|177|178|179|180|181|182|183|184|185|186)$", n190):
        if hasattr(o190, "shape"):
            print(n190, o190.shape)
        elif isinstance(o190, dict):
            print(n190, "dict keys", list(o190)[:12])
        elif callable(o190):
            print(n190, "function")
print("Active edits:", edited())
with open("/work/notes.md", "a") as f190:
    f190.write("\n## Python 190 — inspect an upstream residual correction\n"
               "- Record estimates differ sharply from unadjusted associations; no medical assumptions used.\n"
               "- Investigating a dose-minus-policy correction in the recurrent update, rather than another final-head-only edit.\n")
```