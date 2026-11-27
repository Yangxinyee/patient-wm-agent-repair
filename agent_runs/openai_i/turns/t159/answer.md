The probes suggest glucose and MAP use different unit embeddings. The long diagnostic output hid the recurrent-step result, so I haven’t validated that step yet. I’ll use a compact test of units and empty-action encoding. A near-zero posterior error would identify the correct convention; otherwise the context calculation remains suspect.

```python
import numpy as np, torch
from pathlib import Path

print("=== COMPACT CONTRACTS ===")
for _ev in ["glucose_finger", "map_arterial", "insulin", "norepinephrine"]:
    print(_ev, code_info[_ev])
print("dt139:", dt139[:6].tolist())
print("probe result types:", results138[:2])
print("latest recurrent return:",
      [tuple(x.shape) for x in _result] if isinstance(_result, tuple)
      else type(_result))
if isinstance(_result, tuple):
    for _x in _result:
        _tar = (S[_i,192:960].reshape(4,192)
                if _x.shape[-2:] == (4,192) else S[_i,:192])
        print("latest recurrent RMS:", float(np.sqrt(np.mean((_x.numpy()-_tar)**2))))

obs_ix143 = [int(i) for i in ix139[:32]]
ctx143 = bestctx139[:len(obs_ix143)]
target143 = torch.from_numpy(S[obs_ix143,960:992])
empty_u143 = {
    "zero": torch.zeros(len(obs_ix143),192),
    "encoded": pool137([], "encode_action")[None].expand(len(obs_ix143),-1)
}
post_tests143 = []
for _unit in range(6):
    _ee = torch.stack([
        pool137([atom138(r, unit_index=_unit)
                 for r in bundles[i]["starts"]], "encode")
        for i in obs_ix143
    ])
    for _empty_name, _uu in empty_u143.items():
        _zz = mlp136(torch.cat([ctx143, _ee, _uu], dim=-1), "z_post")[:,:32]
        _rms = float(torch.sqrt(torch.mean((_zz-target143)**2)))
        post_tests143.append((_rms, _unit, _empty_name))
print("posterior unit/empty-action tests:", sorted(post_tests143))

print("=== RECURRENT PROBE WITH MEASUREMENT UNITS ===")
def measurement_atom143(r):
    return atom138(r, unit_index=1 if r["event"].startswith("glucose") else 2)

_i = obs_ix143[0]
_p = int(previous[_i])
_hist = []
_j = _p
while _j >= 0 and len(_hist) < 128:
    _hist.append(_j)
    _j = int(previous[_j])
_hist.reverse()
_mem = torch.stack([
    pool137([measurement_atom143(r) for r in bundles[j]["starts"]
             if r["event"] not in ("insulin", "norepinephrine")]
            + [atom138(r, end=True) for r in bundles[j]["ends"]],
            "encode")
    for j in _hist
])[None]
_ages = torch.tensor([(hours[_i]-hours[j])*3600 for j in _hist],
                     dtype=torch.float32)[None]
_e = pool137([measurement_atom143(r) for r in bundles[_i]["starts"]], "encode")[None]
_h = torch.from_numpy(S[_p,192:960].reshape(1,4,192))
_z = torch.from_numpy(S[_i,960:992])[None]
_dt = torch.tensor([round((hours[_i]-hours[_p])*60)*60], dtype=torch.float32)
for _mode in ["key", "input"]:
    for _empty_name, _uu in empty_u143.items():
        _ret = core_step136(_h, _dt, _e, _uu[:1], _z, _mem, _ages, age_mode=_mode)
        print(_mode, _empty_name,
              [float(np.sqrt(np.mean(
                  (x.numpy()-(S[_i,192:960].reshape(4,192)
                   if x.shape[-2:] == (4,192) else S[_i,:192]))**2)))
               for x in _ret])
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 143 — compact reconstruction test\n"
             "- Testing measurement units and zero versus encoded empty actions "
             "against exposed posterior and recurrent states.\n"
             "- No weight edits active.\n")
```
