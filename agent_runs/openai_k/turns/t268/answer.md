The retained kernel has a core replay with an explicit correction argument and a head evaluator, so a new implementation may not be necessary. I’ll inspect those two boundaries and the dose-feature fits. If they expose the state immediately before the coda and the resulting value probabilities, I can test a small low-rank correction without backpropagation.

```python
import dis, io, numpy as np, torch

def compact_bytecode241(fn, limit):
    buf = io.StringIO()
    for ins in dis.get_instructions(fn):
        if ins.opname in {"RESUME", "CACHE", "NOP", "PRECALL", "PUSH_NULL"}:
            continue
        print(f"{ins.offset}: {ins.opname} {ins.argrepr}", file=buf)
    return buf.getvalue()[:limit]

print("=== CORE REPLAY ===")
print("globals:", core228.__code__.co_names)
print(compact_bytecode241(core228, 3600))

print("\n=== HEAD EVALUATOR ===")
print("globals:", r49_readouts.__code__.co_names)
print(compact_bytecode241(r49_readouts, 4200))

print("\n=== ACTION / TOKEN INTERFACES ===")
for nm in ["action230", "original_token231", "posterior_token154",
           "r179_capture_head", "r233_fn"]:
    fn241 = globals()[nm]
    print(nm, "globals:", fn241.__code__.co_names,
          "locals:", fn241.__code__.co_varnames)

print("\n=== RATE FEATURE FIT OBJECTS ===")
for key241, val241 in r238_features.items():
    print(key241, type(val241).__name__,
          list(val241.keys()) if isinstance(val241, dict)
          else (tuple(val241.shape) if hasattr(val241, "shape")
                else [type(x).__name__ for x in val241]
                if isinstance(val241, (tuple, list)) else ""))

print("\n=== BASE / COUNTERFACTUAL STATES ===")
for nm241 in ["base228", "original_cf231", "r234_capture"]:
    ob241 = globals()[nm241]
    for key241, val241 in ob241.items():
        if isinstance(val241, dict):
            print(nm241, key241, {
                k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                for k, v in val241.items()
            })
        else:
            print(nm241, key241,
                  tuple(val241.shape) if hasattr(val241, "shape")
                  else type(val241).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 241: inspect correction and readout boundaries\n"
            "- Existing core replay accepts an explicit correction; use it rather than rebuilding the model.\n"
            "- Recovering only its injection semantics, head evaluator, and physical-rate feature fits.\n"
            "- Aim: a small low-rank recurrent correction, validated by official factual and rollout passes.\n"
            "- No weights changed. Python runs: 241; forward passes: 17.\n")
```