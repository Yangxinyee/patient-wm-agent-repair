The halving response is already weak at hour 13, before many rounds of feedback, which strengthens the centering concern. I’ll first validate the CPU state-step reconstruction against saved GPU states. Close agreement would make a fixed-history dose test trustworthy; a large discrepancy would require fixing the reconstruction before drawing conclusions.

```python
import numpy as np, torch
from pathlib import Path

_r169_saved_W0 = W0
_r169_saved_s4 = s4
_r169_current = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}
with np.load("/work/fwd_15/states_edited.npz") as _npz:
    _r169_states = {k: _npz[k].copy() for k in _npz.files}

print("state/memory shapes", _r169_states["x"].shape, tuple(wm_e4.shape))
print("posterior tensors", [(n, SHAPES[n]) for n in NAMES if n.startswith("z_post")])
print("atom helper references", action_atom_exact.__code__.co_names)
print("set helper references", wm_set.__code__.co_names)
print("cache columns", list(_r140_cache["df"].columns), "layout", head_layout_spec)

_r169_tests = []
try:
    W0 = _r169_current
    s4 = _r169_states
    _js = np.flatnonzero(
        (s4["has_insulin"] | s4["has_norepinephrine"])
        & (np.r_[False, s4["stay_id"][1:] == s4["stay_id"][:-1]])
    )
    _js = _js[np.linspace(0, len(_js)-1, 8).astype(int)]
    for _j in _js:
        _pred = wm_step_saved(int(_j))
        _truth = torch.as_tensor(s4["x"][_j, 192:960]).reshape(1, 4, 192)
        _ctx = context_from_saved(np.array([_j]), s4)[0]
        _key = (int(s4["stay_id"][_j]), int(round(s4["hour"][_j]*60)))
        _grp = groups4[_key]
        _atoms = [action_atom_exact(r.event, [r.rate], [r.dose])[0]
                  for r in _grp.itertuples(index=False)
                  if r.event in ("insulin", "norepinephrine")]
        _u = wm_set(torch.stack(_atoms), "encode_action").reshape(1, 192)
        _post = wm_mlp(torch.cat([_ctx, wm_e4[_j:_j+1], _u], dim=-1), "z_post")[:, :32]
        _ztruth = torch.as_tensor(s4["x"][_j:_j+1, 960:992])
        _r169_tests.append({
            "j": int(_j), "hour": float(s4["hour"][_j]),
            "state_rmse": float(torch.sqrt(torch.mean((_pred-_truth)**2))),
            "state_max": float(torch.max(torch.abs(_pred-_truth))),
            "posterior_rmse": float(torch.sqrt(torch.mean((_post-_ztruth)**2)))
        })
finally:
    W0 = _r169_saved_W0
    s4 = _r169_saved_s4

print(pd.DataFrame(_r169_tests).to_string(index=False))
```
