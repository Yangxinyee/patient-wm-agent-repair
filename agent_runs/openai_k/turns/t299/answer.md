The glucose-only fallback is intact. The rejected trial adds a separate MAP correction, so I can test smaller combinations without rebuilding the edit. I’ll check their local cross-talk and recover the validation stay IDs. A useful MAP correction should retain its dose response while adding little factual disturbance.

```python
import json
import inspect
from pathlib import Path

def compact265(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return {"type": type(x).__name__, "shape": list(x.shape)}
    if isinstance(x, dict):
        return {str(k): compact265(v) for k, v in list(x.items())[:16]}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x),
                "first": [compact265(v) for v in x[:3]]}
    if isinstance(x, (str, int, float, bool, np.number)):
        return str(x)[:120]
    return type(x).__name__

base265 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
with np.load("/work/recurrent_glucose_valid.npz") as z:
    safe265 = z["b_in"].copy()
current265 = np.asarray(load("core.b_in.weight"), dtype=np.float32)

print("=== MATRIX CHECKS ===")
for name, a in [("glucose correction", r255_CI),
                ("new MAP correction", rr_NC)]:
    a = np.asarray(a)
    print(name, "norm", np.linalg.norm(a))
print("fallback minus base minus CI:",
      np.linalg.norm(safe265 - base265 - np.asarray(r255_CI)))
print("current minus fallback minus NC:",
      np.linalg.norm(current265 - safe265 - np.asarray(rr_NC)))

print("\n=== VALIDATION STAYS ===")
print(json.dumps([int(s) for s in rr_eval_stays]))

print("\n=== CASES AND ROUTES ===")
for name in ["np242", "r245_G", "r245_V", "r245_delta",
             "rr_Ndirection", "rr_Nv", "rr_Nlocal_shifts",
             "rr_Nbase_shifts", "rr_Nsubset_fits", "r244_read_pre",
             "r256_predict", "rr_conservative_map"]:
    if name in globals():
        print(name, compact265(globals()[name]))
        obj = globals()[name]
        if isinstance(obj, types.FunctionType):
            print(" args:", obj.__code__.co_varnames[:obj.__code__.co_argcount],
                  " globals:", obj.__code__.co_names)
            try:
                src = inspect.getsource(obj)
                print(src[:3000])
            except (OSError, TypeError):
                pass

print("\n=== RECORD-BASED IDENTIFICATION ===")
evidence265 = [
    line for line in Path("/work/notes.md").read_text().splitlines()
    if re.search(r"instrument|randomiz|six.lag|impulse|identif|12\.37|4\.383|causal|confound", line, re.I)
]
print("\n".join(evidence265[-65:])[:4800])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 265: verify separable corrections\n"
            "- Checking the glucose fallback and rejected MAP addition against their retained matrices.\n"
            "- Recovering exact local cases and validation stays before attenuation tests.\n"
            "- No weights changed. Python runs: 265; forward passes: 22.\n")
```