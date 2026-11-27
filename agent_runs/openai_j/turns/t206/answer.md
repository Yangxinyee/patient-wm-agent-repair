The reported traceback repeats the earlier input-type failure, so there is still no successful frontier result. I’ll inspect the evaluator’s actual inputs and return format before retrying. This should resolve the mismatch without changing any weights.

```python
import dis
import numpy as np

print("INPUT TYPES")
for name in ["map164_fact", "wm_train145", "wm_tune145", "mc169", "head177_weights"]:
    obj = globals().get(name)
    print(name, type(obj).__name__)
    if isinstance(obj, dict):
        print(" keys:", list(obj.keys()))
        for key in ["inputs", "logits", "target", "nll"]:
            if key in obj:
                value = obj[key]
                print(" ", key, type(value).__name__,
                      tuple(value.shape) if hasattr(value, "shape") else repr(value)[:80])
    elif hasattr(obj, "shape"):
        print(" shape:", tuple(obj.shape))

print("\nEVALUATOR BYTECODE")
groups = []
current = []
line = None
for instruction in dis.get_instructions(head177_eval):
    if instruction.starts_line is not None:
        if current:
            groups.append((line, current))
        line = instruction.starts_line
        current = []
    current.append(instruction.opname + " " + instruction.argrepr)
if current:
    groups.append((line, current))
for line, instructions in groups:
    print(line, " | ".join(instructions))

print("\nDIRECT TUNING EVALUATION")
try:
    result_interface = head177_eval(mc169, head177_weights)
    print("result type:", type(result_interface).__name__)
    if isinstance(result_interface, dict):
        print("keys:", list(result_interface.keys()))
    elif hasattr(result_interface, "shape"):
        print("shape:", tuple(result_interface.shape),
              "mean:", float(np.asarray(result_interface).mean()))
except Exception as exc:
    print(type(exc).__name__, str(exc))

print("\nACTIVE EDITS:", edited())
```