My arterial MAP prediction after an NE bundle is too high by an amount that grows with the NE rate:

| NE rate just given | my error vs tool (arterial) |
|---|---|
| ≤ 0.2 | +0.8 mmHg |
| 0.2–0.5 | +2.5 mmHg |
| 0.5–0.8 | +3.4 mmHg |
| > 0.8 | +3.8 mmHg |

Cuff MAP looks the same. So the real model has an extra NE-rate-dependent term that lowers the MAP prediction when NE is high. That is the "sicker patient" association, not a drug effect.

Hypothesis: measurement heads also read the "filtered view". That view adds `pending_mlp(p)`, gated by `filter_gate([view, p])`, where p encodes the infusions running at prediction time. If true, this pending pathway is a dose-dependent route straight into the MAP head, and the natural place for an edit.

First I need to recover how my atom-embedding and action-encoder helpers are called, so I can build p for a running NE or insulin record. This run inspects their defaults and tries one NE atom.

```python
for fn in ['atom_vec2','atom_vec','encode','zpred','stay_atoms']:
    f_ = globals()[fn]; print(fn, f_.__code__.co_varnames[:f_.__code__.co_argcount], f_.__defaults__)
try:
    av_ = atom_vec2(code_id['norepinephrine'], 0, 2, np.nan, 0.5, 2.4, unit_row, side, None, False)
    print('atom_vec2 ok', av_.shape, float(av_.norm()))
except Exception as ex: print('atom_vec2 err', repr(ex))
print('rate_unit rows', VOCAB.get('categorical', None) if 'categorical' in VOCAB else [k_ for k_ in VOCAB.keys()])
print({k_: (type(v_).__name__, (v_ if isinstance(v_,(int,float,str)) else (list(v_)[:6] if hasattr(v_,'__iter__') else None))) for k_, v_ in [('ACTS',ACTS), ('NUM', NUM), ('names', names)]})
```