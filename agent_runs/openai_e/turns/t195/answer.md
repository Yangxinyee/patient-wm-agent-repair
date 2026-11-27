The factual cost is mostly in glucose, and its mean prediction shifts downward slightly. That suggests a small decoder correction may help, but mean bias alone does not explain NLL. I’ll verify the decoder’s register layout and latent handling before estimating class-level corrections. Wrong alignment would invalidate the calibration.

```python
import dis, collections, torch, numpy as np
from pathlib import Path

def compact_bytecode166(fn, budget=4300):
    groups = []
    current = []
    lineno = None
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None and ins.starts_line != lineno:
            if current:
                groups.append(f"{lineno}: " + " | ".join(current))
            current = []
            lineno = ins.starts_line
        arg = ins.argrepr
        if len(arg) > 100:
            arg = arg[:97] + "..."
        current.append(ins.opname + (" " + arg if arg else ""))
    if current:
        groups.append(f"{lineno}: " + " | ".join(current))
    return "\n".join(groups)[:budget]

print("READOUT125 BYTECODE")
print(compact_bytecode166(readout125, 4400))
print("\nDEFAULT OBJECTS")
for fn in [readout121, readout125]:
    print(fn.__name__)
    for i, obj in enumerate(fn.__defaults__ or ()):
        print(i, type(obj).__name__, getattr(obj, "shape", None),
              repr(obj)[:70] if not hasattr(obj, "shape") else "")

print("\nRELEVANT STATIC OBJECTS")
for k, v in list(globals().items()):
    if re.search(r"(static|transition|centres|source|src|head_meta)", k, re.I):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:6])

print("\nREGISTER-BUILDER BYTECODE")
print(compact_bytecode166(reading_metadata162, 2500))
print("\nVOCAB EXAMPLES")
print("numeric glucose:", numeric162[("glucose_finger", "value")])
print("vocab keys:", list(VOCAB))
print("W0:", type(W0).__name__, len(W0))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 166: inspect exact local decoder implementation\n"
        "Recovered decoder bytecode/defaults and register/static inputs before constructing "
        "training probabilities. Glucose means shift only modestly; calibration must address "
        "probability distributions, not only mean bias. No weights changed. "
        "Python runs: 166; forwards: 22.\n"
    )
```