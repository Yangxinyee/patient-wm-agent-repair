The recovered helpers appear consistent with the architecture, but they still need numerical validation. I’ll check the remaining embedding conventions and test a recurrent step against an exposed state. A close state match would support using local response measurements; a substantial mismatch would require fixing the reconstruction first.

```python
import torch, numpy as np, pandas as pd, dis
from pathlib import Path

torch.set_num_threads(8)
print("=== EMBEDDING CONVENTIONS ===")
print("numeric note:", VOCAB.get("numeric_note"))
print("atom kinds:", VOCAB.get("atom_kinds"))
print("event kinds:", VOCAB.get("event_kinds"))
for _name in NAMES:
    if _name.startswith(("embed.categorical", "embed.kind", "embed.ekind", "z_in")):
        print(_name, SHAPES[_name])
print("best atom probes:", ranked138[:4])
print("best time probe:", best139[:4])
print("probe bundles:")
for _i in ix139[:3]:
    print(_i, sid[_i], hours[_i], repr(bundles[_i])[:650])
print("train/tuning identifiers:")
for _path in ["/data/train.csv", "/data/tuning.csv"]:
    print(_path, pd.read_csv(_path, nrows=2).to_dict("records"))

print("=== RECURRENT HELPER IMPLEMENTATION ===")
dis.dis(core_step136)
print("=== LOCAL STEP PROBE ===")
_i = int(ix139[0])
_p = int(previous[_i])
_hist = []
_j = _p
while _j >= 0 and len(_hist) < 128:
    _hist.append(_j)
    _j = int(previous[_j])
_hist = _hist[::-1]
_emem = []
for _j in _hist:
    _obs = [atom138(r) for r in bundles[_j]["starts"]
            if r["event"] not in ("insulin", "norepinephrine")]
    _obs += [atom138(r, end=True) for r in bundles[_j]["ends"]]
    _emem.append(pool137(_obs, "encode"))
_mem = torch.stack(_emem)[None]
_ages = torch.tensor([(hours[_i]-hours[j])*3600 for j in _hist],
                     dtype=torch.float32)[None]
_obs = [atom138(r) for r in bundles[_i]["starts"]
        if r["event"] not in ("insulin", "norepinephrine")]
_act = [atom138(r) for r in bundles[_i]["starts"]
        if r["event"] in ("insulin", "norepinephrine")]
_obs += [atom138(r, end=True) for r in bundles[_i]["ends"]]
_e = pool137(_obs, "encode")[None]
_u = pool137(_act, "encode_action", zero_empty=True)[None]
_z = torch.from_numpy(S[_i, 960:992])[None]
_h = torch.from_numpy(S[_p, 192:960].reshape(1, 4, 192))
_dt = torch.tensor([(hours[_i]-hours[_p])*3600], dtype=torch.float32)
for _mode in ["key", "input"]:
    try:
        _result = core_step136(_h, _dt, _e, _u, _z, _mem, _ages, age_mode=_mode)
        print("step return:", _mode, type(_result),
              [tuple(x.shape) for x in _result] if isinstance(_result, tuple)
              else tuple(_result.shape))
        if isinstance(_result, tuple):
            for _x in _result:
                _target = S[_i, :192] if _x.shape[-2:] != (4,192) else S[_i,192:960].reshape(4,192)
                print("RMS:", float(np.sqrt(np.mean((_x.numpy()-_target)**2))))
    except Exception as _exc:
        print(type(_exc).__name__, str(_exc)[:250])
```
