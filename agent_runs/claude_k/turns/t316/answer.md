The last run (291) saved the norepinephrine reader in the value head, and it is working as designed:
- **Gate:** it is on only for the two MAP codes (pre-activation +10) and off for every other code (−10).
- **Reader:** its natural sd is 0.69 per unit, and its response to a lag-1 norepinephrine rate change is 1 by construction.
- **Edited tensors:** only the intended ones.

I have not yet measured what this reader does in a rollout or what it costs in factual fit. I now hold four beliefs:
1. Insulin dose information barely reaches the model state at glucose readings, so an insulin reader would add too much noise.
2. Norepinephrine information does reach that state strongly, so a gated reader is viable.
3. The current gain, κ = 1.5, is a first guess.
4. The rough dose-effect targets estimated from the training records, on the 100 tuning stays, are:

| Edit | Target shift | Original model |
|---|---|---|
| Norepinephrine ×2 | +4.6 | +0.34 |
| Norepinephrine ×½ | −2.3 | −0.06 |
| Insulin ×2 | −13.8 | −2.15 |
| Insulin ×½ | +6.9 | +1.23 |

Next I need the exact stay ids of those 100 tuning stays, so the forward requests stay comparable with earlier passes. I also want a helper that rescales κ by scaling only the two paired output columns (units 120 and 110). The output stays short.

What would confirm the plan:
- Only columns 120 and 110 differ from the original.
- They are roughly mirror images (paired).

If other units changed, I will inspect them before any rescaling.

```python
import numpy as np, pandas as pd, json, glob
_tc292 = pd.read_csv('/data/tuning.csv')
_sc292 = [c for c in _tc292.columns if 'stay' in c.lower()][0]
_tids292 = sorted(pd.unique(_tc292[_sc292]).tolist())
_p13 = sorted(glob.glob('/work/fwd_13/dose_edit_*.csv'))
_d13 = pd.read_csv(_p13[0])
_s13 = [c for c in _d13.columns if 'stay' in c.lower()][0]
_ids13 = sorted(pd.unique(_d13[_s13]).tolist())
print('tuning', _sc292, len(_tids292), 'files13', [p.split('/')[-1] for p in _p13], 'n13', len(_ids13), 'first100', _ids13 == _tids292[:100])
print('IDS13=' + json.dumps([int(x) if str(x).lstrip('-').isdigit() else str(x) for x in _ids13], separators=(',', ':')))
_t292 = load('attr_heads.value_bin.2.weight')
_isT292 = hasattr(_t292, 'detach')
_W2b292 = (_t292.detach().cpu().numpy() if _isT292 else np.asarray(_t292)).astype(np.float32).copy()
_o292 = load_original('attr_heads.value_bin.2.weight')
_o292 = (_o292.detach().cpu().numpy() if hasattr(_o292, 'detach') else np.asarray(_o292)).astype(np.float32)
_ax292 = 0 if _W2b292.shape[0] == 17 else 1
_dc292 = np.where(np.abs(_W2b292 - _o292).max(axis=_ax292) > 0)[0]
_c120 = _W2b292[:, 120] if _ax292 == 0 else _W2b292[120]
_c110 = _W2b292[:, 110] if _ax292 == 0 else _W2b292[110]
print('W2', _W2b292.shape, 'changed units', _dc292.tolist(), 'corr120_110 %.3f' % np.corrcoef(_c120, _c110)[0, 1],
      'norms %.3f %.3f maxsum %.4f' % (np.linalg.norm(_c120), np.linalg.norm(_c110), np.abs(_c120 + _c110).max()))
def _set_kappa_ne(kap, base=1.5):
    import torch
    Wk = _W2b292.copy()
    if _ax292 == 0:
        Wk[:, [120, 110]] *= kap / base
    else:
        Wk[[120, 110], :] *= kap / base
    save_edit('attr_heads.value_bin.2.weight', torch.from_numpy(Wk) if _isT292 else Wk)
    return kap
with open('/work/notes.md', 'a') as fh:
    fh.write("\n## Restart 8 (run 292)\n- run292: tuning100 ids (= fwd13 stays) printed; helper _set_kappa_ne(k) rescales value-head out cols 120/110 (saved kappa 1.5). "
             "NEXT: fwd14 dose_edit both tuning100, fwd15 factual both tuning100, rescale kappa toward NE x2 target +4.6 keeping dNLL<.003. Insulin left at original (no viable pathway).\n")
print('edited', edited())
```