The architecture confirms that the two drug channels can affect the recurrent state through `z_in`, without changing the prediction heads. The output was truncated before showing the calibration interfaces. I’ll recover those compactly and capture a local update’s inputs; that should let me test new directions against the validated readouts rather than the earlier unreliable ones.

```python
import sys, inspect, dis
import numpy as np
import torch

def brief148(v, depth=0):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return f"{type(v).__name__}{tuple(v.shape)}"
    if isinstance(v, dict):
        if depth >= 2:
            return "dict keys=" + str(list(v)[:12])
        return {str(k): brief148(x, depth+1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [brief148(x, depth+1) for x in v[:8]]
    if hasattr(v, "coef_"):
        return f"{type(v).__name__}, coef={np.shape(v.coef_)}"
    return repr(v)[:240]

print("=== PAIRED READOUT CONTENTS ===")
for name in ["models145", "diagnostics145"]:
    print(name, brief148(globals().get(name)))

print("\n=== READOUT VARIABLES AND HELPERS ===")
for name, obj in list(globals().items()):
    if any(t in name for t in ["144", "145"]):
        if callable(obj) and hasattr(obj, "__code__"):
            print(name, inspect.signature(obj),
                  "globals:", obj.__code__.co_names,
                  "constants:", repr(obj.__code__.co_consts)[:1100])
        else:
            print(name, brief148(obj))

print("\n=== UPDATE/CANDIDATE DEPENDENCIES ===")
for name in ["r59_core", "repaired_core_local", "fn93"]:
    obj = globals()[name]
    print(name, "globals:", obj.__code__.co_names)
    print("constants:", repr(obj.__code__.co_consts)[:1600])

print("\n=== CANDIDATE RECIPE, COMPACT BYTECODE ===")
lines148 = {}
for instruction in dis.get_instructions(fn93):
    if instruction.starts_line is not None:
        line148 = instruction.starts_line
    lines148.setdefault(line148, []).append(
        instruction.opname.replace("LOAD_", "L:").replace("STORE_", "S:")
        + (" " + instruction.argrepr if instruction.argrepr else "")
    )
for line, parts in lines148.items():
    print(line, " | ".join(parts))

print("\n=== CAPTURE LOCAL UPDATE INPUTS ===")
captures148 = {}
codes148 = {r59_core.__code__, repaired_core_local.__code__}
def profile148(frame, event, arg):
    if event == "return" and frame.f_code in codes148:
        captures148[frame.f_code.co_name] = dict(frame.f_locals)
        captures148[frame.f_code.co_name]["RETURN"] = arg
probe_indices148 = np.asarray(CF89[0]["indices"], dtype=int)[:2]
sys.setprofile(profile148)
try:
    result148 = r59_core(probe_indices148)
finally:
    sys.setprofile(None)
for name, localvars in captures148.items():
    print(name, {k: brief148(v) for k, v in localvars.items()
                 if k not in ["histories", "rows"]})
print("CF89:", [brief148(v) for v in CF89])
print("Saved edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 148: compact interface recovery\n"
            "- Architecture confirms posterior drug channels reach the recurrent state through z_in.\n"
            "- Run 147 output truncated; now capturing actual local-update inputs and "
            "paired readout structure without a full disassembly.\n"
            "- No saved weight changes. Python runs: 148; forward passes: 10.\n")
```