The cached bundle contains four atoms while the reconstructed bundle contains two, so the previous comparison was not between matching bundles. The norepinephrine offset also extrapolates badly, making it unsuitable for an edit as fitted. I’ll verify stay-hour alignment against the exported states; exact identity matches are required before recomputing directions or offsets.

```python
from pathlib import Path
import numpy as np

print("=== BUNDLE IDENTITY CHECK ===")
for _i66 in rr64_np(rr58_mi)[:4].astype(int):
    _sid66 = int(rr64_np(r36_sid)[_i66])
    _hr66 = float(rr64_np(r36_hour)[_i66])
    _rows66 = r36_world.get((_sid66, _hr66))
    print("index", _i66, "cached identity", (_sid66, _hr66),
          "state index", rr64_np(r36_indices)[_i66])
    print("world rows:", repr(_rows66)[:450])
print("rr58 row identities:",
      [(int(rows[0].stay_id), float(rows[0].hour)) for rows in rr58_mrows[:4]])
print("owner bounds:", rr64_np(r36_owners).min(), rr64_np(r36_owners).max(),
      "input length:", len(r36_sid))

print("\n=== STATE EXPORT STRUCTURES ===")
for _dir66 in ("fwd_2", "fwd_4"):
    _file66 = Path("/work") / _dir66 / "states_original.npz"
    if _file66.exists():
        with np.load(_file66) as _npz66:
            print(_dir66, [(k, _npz66[k].shape, str(_npz66[k].dtype))
                          for k in _npz66.files])
            for _key66 in _npz66.files:
                _a66 = _npz66[_key66]
                if _a66.ndim == 1:
                    print(" ", _key66, _a66[:5])

print("\n=== CURRENT DATA / CORE HELPER OBJECTS ===")
for _nm66 in ("r36_source", "r36_row", "r36_key",
              "r33_codes", "r33_numeric", "r33_mlp", "r33_ln"):
    _v66 = globals().get(_nm66)
    if callable(_v66) and hasattr(_v66, "__code__"):
        print(_nm66, _v66.__code__.co_varnames,
              repr(_v66.__code__.co_consts)[:450])
    elif isinstance(_v66, dict):
        print(_nm66, "dict keys", list(_v66)[:10])
    elif isinstance(_v66, (tuple, list)):
        print(_nm66, "length", len(_v66),
              [type(p).__name__ + (str(tuple(p.shape)) if hasattr(p, "shape") else "")
               for p in _v66])
    else:
        print(_nm66, repr(_v66)[:200])

print("selected tuning stay samples:",
      rr64_np(in_selected)[:5], rr64_np(ne_selected)[:5])
with open("/work/notes.md", "a") as _f66:
    _f66.write("\n## Run 66\n"
               "Run 65 showed cached and reconstructed bundles have different atom counts. "
               "Latent direction calibration must be repeated with verified stay-hour alignment. "
               "Unbounded NE expected-dose fit has large extrapolation error and is rejected. "
               "No weights changed. Counts: 66 Python runs, 6 forward passes.\n")
```