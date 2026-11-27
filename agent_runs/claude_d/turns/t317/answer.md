This is a serious problem. On sel1 the current edit costs **+0.0101 nats per reading** (1.52200 vs 1.51191), twice the 0.005 budget. The cost is on glucose_finger (+0.0117) and especially map_arterial (+0.0193); glucose_lab is +0.0013 and map_cuff −0.0015.

My offline estimates for the head edits were about +0.002 in total. For head-only edits the offline head reproduced the tool exactly, so either:
- the E5 encoder edit leaks into factual states, through its small activations on in-range NE atoms (du 0.0575), and hits MAP readings after NE bundles; or
- the head channels cost more on these stays than offline predicted.

This must be fixed before anything else, since the readouts only count if the factual fit holds.

This run breaks the per-reading NLL difference down:
- by event;
- by whether the stay has NE or insulin at all;
- by whether the reading comes before the stay's first NE infusion. Readings before the first NE cannot be touched by the encoder edit.

It also drops the four encoder tensors, leaving E4 (head edits only; E5's encoder is safe in /work/E5enc.npz). The next factual pass then measures the encoder's contribution directly.

If the extra MAP cost sits in NE stays after the first NE, the encoder leakage is the culprit. If it appears pre-NE too, or in non-NE stays, the head channels are.

```python
import numpy as np, pandas as pd
fo = pd.read_csv('/work/fwd_15/factual_original.csv'); fe = pd.read_csv('/work/fwd_15/factual_edited.csv')
print(fo.shape, list(fo.columns)); print(fo.head(2).to_string())
nc = [c for c in fo.columns if 'nll' in c.lower()][0]; mc = [c for c in fo.columns if 'mean' in c.lower()]
assert (fo.stay_id.values == fe.stay_id.values).all()
fo['d'] = fe[nc].values - fo[nc].values
if mc: fo['dm'] = fe[mc[0]].values - fo[mc[0]].values
ne_first = {int(s): (int(np.argmax(NOR_U[i] > 0)) if np.nansum(NOR_U[i]) > 0 else 99) for s, i in sid2i.items()}
ins_st = {int(s) for s, i in sid2i.items() if np.nansum(INS_U[i]) > 0}
fo['ne_st'] = fo.stay_id.map(lambda s: ne_first.get(int(s), 99) < 99); fo['ins_st'] = fo.stay_id.isin(ins_st)
hc = 'hour' if 'hour' in fo.columns else None
if hc: fo['postNE'] = fo[hc] > fo.stay_id.map(lambda s: ne_first.get(int(s), 99))
print(fo.groupby('event')['d'].agg(['size','mean']).round(4).to_string())
gcols = ['event','ne_st'] + (['postNE'] if hc else [])
print(fo.groupby(gcols)['d'].agg(['size','mean','sum']).round(4).to_string())
print(fo[fo.event=='glucose_finger'].groupby('ins_st')['d'].agg(['size','mean']).round(4).to_string())
if mc: print('mean shift by event', fo.groupby('event')['dm'].agg(['mean','std']).round(3).to_string())
for n_ in ['encode_action.mlp1.0.bias','encode_action.mlp1.0.weight','encode_action.mlp1.2.bias','encode_action.mlp1.2.weight']: drop_edit(n_)
print('now edited:', edited())
```