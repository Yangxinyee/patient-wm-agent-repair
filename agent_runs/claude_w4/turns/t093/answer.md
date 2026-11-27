The summary readout is the pooled mean over the reached stay-hours, e.g. −0.348 for insulin x2. That matches the pooled CSV mean exactly.

Next I compute what my estimated impulse responses imply as the true readouts on the same 100 tuning stays and the same reached stay-hours:
- the finger response `hF` for insulin;
- the arterial response `hA` for norepinephrine;
- the shift at hour t is Σ h(l)·Δrate(t−l), counting only doses from hour 12 on.

This gives the calibration targets for the forward-tool readouts. I will also check that my own definition of `reached` matches the tool's.

```python
def hourly_rates(df, drug, sids):
    d_ = df[(df.event==drug) & df.stay_id.isin(sids)].copy()
    d_['s'] = np.floor(d_.hour + 1e-6).astype(int)
    R = np.zeros((len(sids), 26)); pos = {s:i for i,s in enumerate(sids)}
    for sid_, s_, r_ in zip(d_.stay_id, d_.s, d_.rate):
        if 0 <= s_ < 26: R[pos[sid_], s_] += r_
    return R
def truth(df, drug, irf, mult, start=12, sids=None, hours=range(13,25)):
    R = hourly_rates(df, drug, sids)
    dR = np.where(np.arange(26)[None,:] >= start, (mult-1)*R, 0.0)
    vals, keys = [], []
    for i in range(len(sids)):
        for t in hours:
            if not (R[i, start:t] > 0).any(): continue
            sh = sum(irf[l-1]*dR[i, t-l] for l in range(1, 13) if t-l >= 0)
            vals.append(sh); keys.append((sids[i], t))
    return np.array(vals), keys
s100 = list(sel100)
for drug, irf, ed2, edh in [('insulin', hF, 'insulin_x2', 'insulin_half'), ('norepinephrine', hA, 'ne_x2', 'ne_half')]:
    v2, k2 = truth(tu, drug, irf, 2.0, sids=s100); vh, kh = truth(tu, drug, irf, 0.5, sids=s100)
    reached = set(map(tuple, de1[(de1.edit==ed2) & de1.reached][['stay_id','hour']].values.tolist()))
    print(drug, 'true x2 %.3f  x1/2 %.3f  n=%d  reached-match %d/%d' % (v2.mean(), vh.mean(), len(v2), len(set(k2) & reached), len(reached)))
```