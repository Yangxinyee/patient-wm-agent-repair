The insulin encoder is nearly linear over its usual dose range, even with the tested end-event variants. That makes it an unlikely explanation for the large asymmetry. The norepinephrine check stopped on an invalid synthetic unit index, so I’ll complete it and recover the earlier strength-trial rollouts. If linear encoding holds and increased readout strength helped little, a modest persistence trial will be the next test.

```python
import numpy as np
import pandas as pd
import torch

print("=== NOREPINEPHRINE FINITE ENCODER RESPONSE ===")
_rates263 = np.array([0., .025, .05, .1, .2, .4, .8], np.float32)
_atoms263 = drug_atom205("norepinephrine", _rates263, weights=W262)
_rows263 = {"rate": _rates263}
for _label263, _end263 in (
    ("WORLD", None),
    ("plus_END_ek2", end_atom262("norepinephrine", 2, False)),
    ("plus_END_ek0", end_atom262("norepinephrine", 0, False)),
):
    _bundles263 = _atoms263[:, None, :] if _end263 is None else torch.stack(
        (_atoms263, _end263.expand(len(_rates263), -1)), dim=1)
    _u263 = set205(_bundles263, "encode_action", weights=W262)
    _rows263[_label263] = (_u263[:, 15] - _u263[:, 66]).detach().numpy()
print(pd.DataFrame(_rows263).round(7).to_string(index=False))

print("\n=== ROLLOUT SUMMARIES ===")
for _k263 in (14, 17, 18, 20, 21, 23, 24):
    _path263 = Path(f"/work/fwd_{_k263}/dose_edit_edited.csv")
    if not _path263.exists():
        continue
    _df263 = pd.read_csv(_path263)
    print("forward file", _k263, "columns", _df263.columns.tolist())
    _editcol263 = next((c for c in _df263.columns if c in ("edit", "intervention")), None)
    _shiftcol263 = next((c for c in _df263.columns if c in ("shift", "mean_shift", "delta")), None)
    if _editcol263 and _shiftcol263:
        print(_df263.groupby(_editcol263)[_shiftcol263].mean().round(4).to_dict())
    else:
        print(_df263.head(2).to_string(index=False))

print("\n=== SAVED STATE SCALE ===")
_s263 = np.load("/work/fwd_15/states_edited.npz")
_h263 = _s263["x"][:, 192:960].reshape(-1, 4, 192)
_dt263 = np.zeros(len(_h263), np.float64)
_sid263, _hour263 = _s263["stay_id"], _s263["hour"]
_same263 = _sid263[1:] == _sid263[:-1]
_dt263[1:] = np.where(_same263, np.maximum(0, np.diff(_hour263)) * 3600, 0)
_prev263 = np.concatenate((np.zeros_like(_h263[:1]), _h263[:-1]), axis=0)
_prev263[~np.r_[False, _same263]] = 0
_flow263 = np.exp(_np259("core.flow.a"))
_hf263 = _prev263 * np.exp(-_dt263[:, None, None] * _flow263[None, None, :])
_sig263 = _hf263.std(axis=2).mean(axis=1)
_good263 = (_hour263 >= 6) & (_sig263 > 1)
print("post-flow ordinary scale percentiles",
      np.percentile(_sig263[_good263], [5, 25, 50, 75, 95]).round(3))
print("IDs", _sid263[:3], "hours", _hour263[:8])
print("unit table shapes", SHAPES["embed.categorical.unit.weight"],
      SHAPES["embed.categorical.rate_unit.weight"])

with open("/work/notes.md", "a") as _f263:
    _f263.write(
        "\n## Runs 261–263: finite transport audit\n"
        "- Current insulin WORLD encoder is essentially linear through rate 8; saturation starts near 11.5.\n"
        "- Plausible END variants add an offset but do not reproduce the large doubling/halving asymmetry.\n"
        "- Synthetic END semantics remain unverified; no conclusion rests on them alone.\n"
        "- Completed NE audit and recovered prior rollout summaries; no weights changed.\n"
    )
```