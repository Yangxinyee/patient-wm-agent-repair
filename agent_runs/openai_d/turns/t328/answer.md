Earlier stronger norepinephrine edits exceeded the factual budget, so a large increase is not justified. I’ll check the interfaces for a small, reversible increase from 0.35 to 0.40, leaving insulin unchanged. I would retain it only if all-400 factual validation keeps the same conservative safety margin and rollout validation shows improvement; otherwise I’ll restore the current checkpoint.

```python
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

print("=== FIRST200 POOLED ROLLOUTS ===")
print(_roll272.groupby(["model", "edit"])["shift"]
      .agg(["count", "mean"]).to_string())

print("=== CANDIDATE FUNCTION SIGNATURES ===")
for _name in ("r246_candidate", "n262_logits", "n262_table",
              "r246_loss", "r245_summary"):
    _fn = globals()[_name]
    print(_name, inspect.signature(_fn))
    print("referenced names:", _fn.__code__.co_names)
    print("arguments/local names:", _fn.__code__.co_varnames[:12])

print("=== SAVED SCREEN CANDIDATE STRUCTURE ===")
for _key in ((0.90, 0.35, 4.0), (0.90, 0.40, 4.0)):
    _obj = n262_candidates[_key]
    print(_key, type(_obj).__name__)
    if isinstance(_obj, dict):
        for _k, _v in _obj.items():
            print(" ", _k, type(_v).__name__,
                  getattr(_v, "shape", str(_v)[:90]))
    elif isinstance(_obj, (tuple, list)):
        for _j, _v in enumerate(_obj):
            print(" ", _j, type(_v).__name__,
                  list(_v)[:8] if isinstance(_v, dict)
                  else getattr(_v, "shape", str(_v)[:90]))

print("=== SMALL AMPLITUDE SCREEN ===")
print(n262_screen[
    (n262_screen["ins"] == 0.90) &
    (n262_screen["cancel_scale"] == 4.0)
].to_string(index=False))

print("=== CURRENT CHECKPOINT EXACT BACKUP CHECK ===")
with np.load("/work/candidate90_35_bin80x4.npz") as _z:
    for _k in _z.files:
        _cur = load(_k)
        if hasattr(_cur, "detach"):
            _cur = _cur.detach().cpu().numpy()
        print(_k, "max_abs_difference", float(np.max(np.abs(_cur - _z[_k]))))

print("=== TRANSITION TABLE CODING ===")
print("codes", {k: r257_codes[k] for k in
                ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff")})
print("transition shape", SHAPES["embed_transition.value_bin.weight"])
print("table sample", [(k, np.shape(v))
                       for k, v in list(r257_table.items())[:3]])

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Run 273: bounded stronger-NE trial planning\n"
        "No weights changed. Considering insulin .90 / NE .40 with the same "
        "training-only shrink80-scale4 cancellation. Precommitment: retain "
        "only if all400 paired factual validation has upper95 below .005 "
        "and exact-checkpoint rollout improves NE without materially harming "
        "insulin. Otherwise restore candidate90_35_bin80x4.npz, already "
        "validated on all400 factual records and two rollout cohorts.\n"
    )
```
