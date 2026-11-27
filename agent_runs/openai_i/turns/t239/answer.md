Both private response columns are currently zero, so the neutral edit has not yet added any drug effect. The records also show multi-hour effects, which means matching a one-hour slope alone would be insufficient. Next I’ll check coordinate scaling and replay inputs; then I can measure response directions without guessing their magnitude or persistence.

```python
import dis
import inspect
import re
from pathlib import Path

print("FULL LAGGED RECORD EVIDENCE")
parts213 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for s213 in parts213:
    if s213.startswith("## Lagged-effect") or s213.startswith("## Python 149"):
        print(s213[:4800])

print("\nPRIVATE COORDINATE CONTRACTS")
for name213 in ["private_indices209", "private_dims196", "private_z209",
                "orig_z209", "innov194", "policy_pred194", "rates194",
                "presence194", "feature_coef201", "feature_pred201",
                "coef209", "X209", "hidden196"]:
    if name213 in globals():
        obj213 = globals()[name213]
        print(name213, type(obj213).__name__, getattr(obj213, "shape", None))
        if isinstance(obj213, (list, tuple)):
            print("value", obj213)
        elif hasattr(obj213, "shape") and np.size(array196(obj213)) <= 16:
            print(array196(obj213))

if "private_z209" in globals():
    zp213 = array196(private_z209)
    print("private coordinate statistics",
          np.column_stack([zp213[:, [16, 12]].mean(0),
                           zp213[:, [16, 12]].std(0),
                           np.min(zp213[:, [16, 12]], axis=0),
                           np.max(zp213[:, [16, 12]], axis=0)]))
    if "innov194" in globals():
        in213 = array196(innov194)
        print("innovation shape", in213.shape)
        if in213.shape == (len(zp213), 2):
            print("coordinate minus innovation RMS",
                  np.sqrt(np.mean((zp213[:, [16, 12]] - in213)**2, axis=0)))

print("\nREPLAY ARGUMENTS")
print("args count", len(args_cache182[0]))
for i213, obj213 in enumerate(args_cache182[0]):
    print(i213, type(obj213).__name__, getattr(obj213, "shape", None))
print("contrast list example", type(contrasts182[0]).__name__)
if isinstance(contrasts182[0], dict):
    print("contrast keys", list(contrasts182[0]))
elif isinstance(contrasts182[0], (tuple, list)):
    print([(type(v213).__name__, getattr(v213, "shape", None))
           for v213 in contrasts182[0]])

print("\nCORE AND LINEAR CONTRACTS")
for name213 in ["core_step136", "linear136", "norm136", "batch_q182"]:
    fn213 = globals().get(name213)
    if callable(fn213):
        print(name213, inspect.signature(fn213),
              "locals", fn213.__code__.co_varnames,
              "globals", fn213.__code__.co_names)

print("\nSMALL HELPER BYTECODE")
for name213 in ["linear136", "batch_q182"]:
    print(name213)
    print(" | ".join(
        f"{ins213.opname}:{ins213.argrepr}"
        for ins213 in dis.Bytecode(globals()[name213])
        if ins213.opname not in ["RESUME", "CACHE"]
    )[:2600])

print("\nACTION/CORE HELPER NAMES")
print(" ".join(sorted(
    n213 for n213, v213 in list(globals().items())
    if callable(v213) and re.search(r"action|embed|prelude|core|capture|build", n213, re.I)
)))

with open("/work/notes.md", "a") as f213:
    f213.write(
        "\n## Python 213 — verify scaling and multi-hour calibration requirements\n"
        "- Neutral private columns remain zero.\n"
        "- Recovering innovation units and replay inputs before finite-difference response measurements.\n"
        "- One-hour calibration will require separate rollout/persistence validation.\n"
    )
```