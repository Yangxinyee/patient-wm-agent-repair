The readouts use 192-dimensional state features and last-reading interactions, but the output was again truncated before the update details. I’ll print only the missing pieces now. The key checks are whether the captured local update reconstructs the official state and whether I can reproduce the readout features exactly; both are needed before trusting a new routing direction.

```python
import dis
import numpy as np
import torch

print("=== CAPTURED UPDATE ===")
for name, localvars in captures148.items():
    print(name, {k: (
        tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor))
        else type(v).__name__
    ) for k, v in localvars.items() if k not in ["histories", "rows"]})
    ret = localvars.get("RETURN")
    print("return:", brief148(ret))

print("\n=== UPDATE DEPENDENCIES ===")
for name in ["r59_core", "repaired_core_local"]:
    obj = globals()[name]
    print(name, obj.__code__.co_names)
    print("constants:", repr(obj.__code__.co_consts))

print("\n=== CANDIDATE RECIPE ===")
line149 = 0
parts149 = []
for inst in dis.get_instructions(fn93):
    if inst.starts_line is not None and parts149:
        print(line149, " ".join(parts149))
        parts149 = []
    if inst.starts_line is not None:
        line149 = inst.starts_line
    if inst.opname not in ["RESUME", "PRECALL", "CACHE", "PUSH_NULL"]:
        parts149.append(inst.opname.replace("LOAD_", "L").replace("STORE_", "S")
                        + ":" + inst.argrepr)
if parts149:
    print(line149, " ".join(parts149))

print("\n=== READOUT FEATURE RECOVERY ===")
print("145 variable names:",
      [n for n in globals() if "145" in n and n not in
       ["models145", "diagnostics145", "brief148"]])
for event, m in models145.items():
    ids = m["ids"]
    ao = m["ao"]
    base = m["base"]
    print(event, m["diagnostic"],
          "base-match", float(np.max(np.abs(ao[:, :11] - base))),
          "tail-sample", np.round(ao[:3, -4:], 4).tolist())
    for variant, values in V144o.items():
        v = values[ids]
        print(" ", variant,
              "block11-error", float(np.mean(np.abs(ao[:, 11:203]-v))),
              "block12-error", float(np.mean(np.abs(ao[:, 12:204]-v))))
print("views144 constants:", views144.__code__.co_consts)

print("\n=== COUNTERFACTUAL INPUTS AND RECORD TABLE ===")
for cf in CF89:
    print({k: (tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor))
               else repr(v)[:100]) for k, v in cf.items()})
print("rr35_table columns:", list(rr35_table.columns))
print(rr35_table.head(5).to_string(index=False))
print("train state metadata:", list(zip(
    rr33_stays[probe_indices148], rr33_hours[probe_indices148])))
print("edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 149: targeted update/feature recovery\n"
            "- Paired readouts use state features with last-reading interactions.\n"
            "- Previous diagnostic output again truncated; printing only missing "
            "update context, candidate recipe, and feature-block checks.\n"
            "- No saved changes. Python runs: 149; forward passes: 10.\n")
```