The notes identify earlier threshold analyses, and the kernel still contains the gate configuration and checkpoints. I’ll retrieve the actual estimates and the compact helper definitions next. This will clarify which drug effects are supported by the records and which edit components can be changed safely.

```python
print("=== IDENTIFICATION AND GATE NOTES ===")
wanted_headers = (
    "Recovery and threshold identification",
    "Sustained insulin target",
    "Dynamic identification; Python 10",
    "Activity-gated scale-8 trial",
    "Gate-threshold trial"
)
for section in note_sections:
    header = section.splitlines()[0] if section.strip() else ""
    if any(w in header for w in wanted_headers):
        print(section[:1250])

print("\n=== HELPER NAMES AND ARGUMENTS (NO DEFAULT TENSORS) ===")
for name, obj in sorted(recovery_globals.items()):
    if name.startswith("_") or not inspect.isfunction(obj):
        continue
    code = getattr(obj, "__code__", None)
    if code is not None:
        print(name, code.co_varnames[:code.co_argcount])

print("\n=== GATE CONFIGURATIONS ===")
for name in ("activity_gate_config", "gate035_config", "activity_fit"):
    obj = globals().get(name)
    print(name)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if hasattr(value, "shape"):
                a = as_np_restore(value)
                print(" ", key, a.shape, "norm", float(np.linalg.norm(a)))
            else:
                print(" ", key, repr(value)[:150])

print("\n=== OTHER RELEVANT GLOBAL NAMES ===")
terms = ("select", "graft", "unit", "index", "idx", "contrast", "r3",
         "rho", "rd", "panel", "ids", "coef", "config")
names = [
    n for n in recovery_globals
    if not n.startswith("_") and any(t in n.lower() for t in terms)
]
print(", ".join(sorted(names))[:1600])
```