The stronger edit produced insulin shifts of −10.25 and +5.05, substantially closer to the provisional targets; norepinephrine remains unrepaired. I’ll preserve this candidate and recover the local evaluation machinery. That will let me assess further insulin amplification without losing the fallback. Extra strength is useful only if it closes the rollout gap while retaining factual-fit margin.

```python
import inspect
import re
import numpy as np

ns115 = dict(globals())
r115_strength2 = {n: np.array(load(n), copy=True) for n in edited()}
print("=== RECENT CONSTRUCTION VARIABLES ===")
for k in sorted(ns115):
    if re.search(r"(99|10[0-9]|11[0-2])", k):
        v = ns115[k]
        shape = getattr(v, "shape", None)
        if isinstance(v, dict):
            detail = "keys=" + repr(list(v)[:18])
        elif isinstance(v, (list, tuple)):
            detail = f"len={len(v)}"
        elif shape is None and not callable(v):
            detail = repr(v)[:100]
        else:
            detail = ""
        print(k, type(v).__name__, shape, detail)

print("\n=== LOCAL EVALUATOR ===")
print("signature", inspect.signature(local104))
print("names", local104.__code__.co_names)
print("locals", local104.__code__.co_varnames)
try:
    print(inspect.getsource(local104))
except (OSError, TypeError):
    print("Source unavailable; function remains callable.")

print("\n=== STRENGTH-1 BACKUP ===")
with np.load("/work/insulin_candidate_v1.npz") as f115:
    print("keys", f115.files)
    for n in f115.files:
        if n in r115_strength2:
            a115 = np.asarray(f115[n])
            d115 = r115_strength2[n] - a115
            changed115 = (np.flatnonzero(np.max(np.abs(d115), axis=0) > 1e-8).tolist()
                          if d115.ndim == 2 else
                          np.flatnonzero(np.abs(d115) > 1e-8).tolist())
            print(n, "max_difference", float(np.max(np.abs(d115))),
                  "changed_columns_or_entries", changed115[:30])

with open("/work/notes.md", "a") as f115:
    f115.write(
        "\n## Restart / Runs 113–115 and Forward 12–13\n"
        "Strength-2 insulin candidate (hash 43d7176a64): paired factual NLL "
        "on tuning stays 2020000100..199 is 1.62310 vs 1.62044, delta +0.00266. "
        "On tuning stays 2020000000..99 its dose shifts are insulin x2 -10.248, "
        "half +5.053; NE x2 -0.513, half -0.604. "
        "Provisional record-derived targets for those rollout hours remain "
        "-13.599/+6.800 and +4.746/-2.373. "
        "Saved current strength-2 tensors in r115_strength2. "
        "Further amplification requires factual-budget validation; NE remains unrepaired.\n"
    )
```
