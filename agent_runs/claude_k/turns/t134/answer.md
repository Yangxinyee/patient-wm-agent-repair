End atoms carry no attributes. The best convention is a bare end atom routed through the measurement encoder, with u as the empty-set encoding when a bundle has no new records. That gives a posterior error of 0.0035 on all action bundles and is essentially exact for new-record atoms.

This matters for the design. The model receives each action rate exactly once, through the new-record atom at t:05. So any effect beyond one hour has to be carried by the state itself.

To build a channel I need the whole step, and the cross-attention memory holds the e vectors of earlier reading bundles. So I'll next verify the measurement encoder the same way: on reading bundles (hours ≥ 1, no admissions), compare z_post([ctx, e, u_empty]) with the tool's z. Here e is built from glucose/MAP atoms with value z-scores and strict bins. An error near 0.001 confirms the measurement-atom conventions.

```python
VAL = {e['code']: e for e in VOCAB['numeric'] if e['attribute'] == 'value'}
CID = {'glucose_finger': 6, 'glucose_lab': 10, 'map_arterial': 7, 'map_cuff': 5}
trS99 = tr[tr.stay_id.isin([int(s) for s in set(sidS)])]
rd = trS99[trS99.event.isin(CID.keys())].groupby(['stay_id', 'hour'])
def meas_atoms(s, h):
    try: g_ = rd.get_group((s, h))
    except KeyError: return []
    V = []
    for _, r_ in g_.iterrows():
        nm = r_.event; ed = np.array(VAL[nm]['fine_bin_edges']); x = r_.value
        V.append(atom_vec(CID[nm], kind=0, ekind=1, vbin=int(1 + (ed < x).sum()), zv=float(np.clip((x - VAL[nm]['median']) / VAL[nm]['iqr'], -10, 10)), hv=1.0))
    return V
jsR = [j for j in range(1, len(hrS)) if sidS[j - 1] == sidS[j] and abs(hrS[j] - round(hrS[j])) < 1e-6 and hrS[j] >= 1][:300]
errs = []
for j in jsR:
    V = meas_atoms(int(sidS[j]), float(hrS[j])); 
    if len(V) != S2['n_atoms'][j]: continue
    ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
    errs.append(np.abs(xs[j, 960:992] - zmean(ctx, setenc(np.stack(V), 'encode'), U_EMPTY)).mean())
print('reading bundles n=%d mean err %.4f max %.4f' % (len(errs), np.mean(errs), np.max(errs)))
```