The backup failed because `load()` returns a Torch tensor, not a NumPy array; no weights changed. I’ll retry with tensor cloning and recover the fitting interfaces. The validated candidate remains intact.

```python
from pathlib import Path
import inspect, re

backup237 = {n: load(n).clone() for n in edited()}
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nLIVE FITTING OBJECTS:")
for n in sorted(list(globals())):
    if re.search(r"(179|19[0-9]|20[0-9]|21[0-9]|22[0-9]|23[0-6])$", n):
        v = globals()[n]
        if callable(v):
            try:
                print(n, "FUNCTION", inspect.signature(v))
            except Exception:
                print(n, "CALLABLE")
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, v.shape)
        elif isinstance(v, dict):
            print(n, "dict keys:", list(v.keys())[:30])
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v))
        else:
            print(n, type(v).__name__, repr(v)[:180])

print("\nRECENT FITTING FUNCTION SOURCES:")
for n in sorted(list(globals())):
    if callable(globals()[n]) and re.search(r"(23[0-6])$", n):
        try:
            print(n, inspect.getsource(globals()[n]))
        except Exception:
            f = globals()[n]
            if hasattr(f, "__code__"):
                print(n, "names:", f.__code__.co_names, "vars:", f.__code__.co_varnames)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 237–238 / gated head validated, partial causal repair\n"
            "pair4 lambda=.1 etaI=etaNE=.3: GPU factual NLL 1.63421 vs 1.63164, delta +.00257 "
            "on 100 tuning stays. Paired rollout shifts I x2 -7.878, I half +3.322, "
            "NE x2 +2.231, NE half -1.038 (original -3.680,+2.018,+.465,-.871). "
            "All four improve toward records-derived estimates but remain too small. "
            "backup237 preserves this seven-tensor validated candidate. Run237 stopped "
            "during backup before making changes; corrected using tensor.clone().\n")
```