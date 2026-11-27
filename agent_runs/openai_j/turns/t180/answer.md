The rollout supports the insulin repair: doubling now gives −11.07 and halving +5.89, close to the record-derived estimates. Norepinephrine remains substantially under-responsive. I’ll preserve this candidate and recover the fitting machinery for a MAP-only correction; that extension must improve norepinephrine without sacrificing the insulin gain or factual-fit limit.

```python
from pathlib import Path
import inspect, re, numpy as np, pandas as pd

best_insulin_v1 = {n: load(n).clone() for n in edited()}
np.savez("/work/best_insulin_v1.npz",
         **{n: t.cpu().numpy() for n, t in best_insulin_v1.items()})

print("=== ROLLOUT 16 ===")
for _m in ["original", "edited"]:
    _d = pd.read_csv(f"/work/fwd_16/dose_edit_{_m}.csv")
    print(_m, _d[_d.reached].groupby("edit")["shift"].mean().to_dict())

print("\n=== STOCHASTIC FIT/CANDIDATE INVENTORY ===")
for _k, _v in list(globals().items()):
    if _k.startswith("_") or not re.match(r"(wm_|trial15[89]|r12[89]|r13[01])", _k):
        continue
    if isinstance(_v, (np.ndarray, pd.DataFrame)):
        print(_k, type(_v).__name__, _v.shape)
    elif hasattr(_v, "shape"):
        print(_k, type(_v).__name__, tuple(_v.shape))
    elif isinstance(_v, dict):
        print(_k, "dict", [(str(k), tuple(v.shape) if hasattr(v, "shape")
                          else type(v).__name__) for k, v in list(_v.items())[:16]])
    elif callable(_v):
        print(_k, "function", inspect.signature(_v))
    elif isinstance(_v, (int, float, str, list, tuple)):
        print(_k, str(_v)[:220])

print("\n=== FIT FUNCTION DETAILS ===")
for _name in ["wm_mc145", "wm_mc_raw146", "wm_loss151",
              "wm_moments151", "wm_rawloss152", "trial158_audit"]:
    _fn = globals().get(_name)
    if callable(_fn):
        print("\n", _name, inspect.signature(_fn))
        try:
            print(inspect.getsource(_fn)[:1800])
        except (OSError, TypeError):
            print("referenced names:", _fn.__code__.co_names)
            print("locals:", _fn.__code__.co_varnames)
            print("constants:", str(_fn.__code__.co_consts)[:600])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Forward 16 and run 162: working insulin repair\n"
             "- GPU rollout, first 100 tuning: insulin x2 -11.073 (original -3.680); "
             "half +5.894 (original +2.018).\n"
             "- Record-derived geometric-kernel estimates on these stays: "
             "x2 -10.5288; half +5.2644, with persistence uncertainty.\n"
             "- NE x2 +0.507, half -0.819: little change, still below "
             "record-based +3.8084/-1.9042.\n"
             "- Preserved four-tensor working edit as best_insulin_v1 and "
             "/work/best_insulin_v1.npz. Factual delta +0.00307 on first 100.\n"
             "- Next: independent factual validation and cautious MAP-only extension.\n")
```