The vocab layout is now clear. Each `numeric` entry has a code's median, IQR (z = (x − median)/IQR) and 16 fine-bin edges, and `value_bin_centres` gives per-code bin centres.

I'll now run `gen_samples` on the 100 setA tuning stays and push the vectors through the value head:
- GELU MLP plus the transition row for (code, last bin);
- softmax, then Σ p·centre.

I'll match the result to the tool's `predicted_mean` in `fwd_2` by (stay, hour, code). I'll report MAE for finger glucose and arterial MAP, with and without renormalising away the "no value" class. Small MAE (≲1) validates local factual residuals. A large one means the vector is not exactly the head input, or the z-averaging matters.

```python
cid = {c['name']: c['id'] for c in VOCAB['codes']}
NUMV = {(d['code'], d['attribute']): d for d in VOCAB['numeric']}
W1 = load('attr_heads.value_bin.0.weight').numpy(); b1 = load('attr_heads.value_bin.0.bias').numpy()
W2 = load('attr_heads.value_bin.2.weight').numpy(); b2 = load('attr_heads.value_bin.2.bias').numpy()
TR = load('embed_transition.value_bin.weight').numpy()
from scipy.special import erf
gel = lambda z: 0.5*z*(1+erf(z/np.sqrt(2)))
def lastbin_of(code, v):
    if not np.isfinite(v): return 0
    return 1 + int(np.searchsorted(NUMV[(code,'value')]['fine_bin_edges'], v, side='left'))
def head_mean(rows, renorm=False):
    X_ = np.stack([r[8] for r in rows]); cs = [r[0] for r in rows]
    lg = gel(X_@W1.T + b1)@W2.T + b2 + TR[[cid[c]*17 + lastbin_of(c, r[3]) for c, r in zip(cs, rows)]]
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    C = np.array([np.r_[0.0, VOCAB['value_bin_centres'][c]][-17:] if len(VOCAB['value_bin_centres'][c]) == 16 else VOCAB['value_bin_centres'][c] for c in cs])
    mu = (p*C).sum(1); 
    return (mu/(1-p[:,0])) if renorm else mu, p
print('centre list len', len(VOCAB['value_bin_centres']['glucose_finger']))
grp_tu = {s: g_ for s, g_ in tu.groupby('stay_id', sort=False)}
rowsA = gen_many(grp_tu, list(setA_ids))
dfA = pd.DataFrame({'stay_id': [r[1] for r in rowsA], 'hour': [r[2] for r in rowsA], 'event': [r[0] for r in rowsA]})
for rn in [False, True]:
    mu, _ = head_mean(rowsA, renorm=rn); dfA['mine'] = mu
    mg_ = dfA.merge(f2, on=['stay_id','hour','event'], how='inner')
    print('renorm', rn, 'matched', len(mg_), {ev: round(float(np.abs(g_.mine - g_.predicted_mean).mean()), 3) for ev, g_ in mg_.groupby('event')})
```