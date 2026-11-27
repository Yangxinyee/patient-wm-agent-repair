The decoders are near-exact:
- insulin rate: R² 0.9992, error sd 0.04 u/h;
- NE rate: R² 0.998, error sd 0.019 mcg/kg/min;
- both output about 0 on the empty token.

Their noise is negligible against the innovations. The model's own dose direction in the residual stream is large (79.6 per u/h for insulin, 277 per mcg/kg/min for NE), and head 2 can reproduce it with cosine 0.93. Amplifying that direction around 10–30× could swamp the slots before the LayerNorm. So I want to measure how a head-2 injection actually moves teacher-forced predictions, and how far it persists to lags 2–4.

This run times the local harness (exact core plus approximate head) on 100 training stays. It measures the lag profile of a small injection along the model's own direction, for both drugs. It also checks that the decoders still read doubled doses linearly; the ×2 readout needs that, and NE at twice the dose reaches bins never seen in training.

```python
cand = [s for s in selT if (OT['ins'][idxT[s]]>0).sum()>=3 or (OT['ne'][idxT[s]]>0).sum()>=3]
H100 = cand[:100]; print('cand', len(cand))
KV0 = Wo['core.prelude.0.attn.kv.weight'].copy()
def run_rows(stays):
    rows = [r for s in stays for r in gen_samples(G3[s], pert=False)]; return rows, np.asarray(head_mean(rows))
t0 = time.time(); rows0, mu0 = run_rows(H100); print('rows', len(rows0), 'code ex', rows0[0][:3], 'secs %.1f' % (time.time()-t0))
codes = np.array([str(r[0]) for r in rows0]); ii = idxT[[r[1] for r in rows0]].values; cc = np.round([r[2] for r in rows0]).astype(int)+1
def lagX(drug, K=4):
    M = OT[drug]; return np.column_stack([np.where(cc-k>=0, M[ii, np.maximum(cc-k,0)], 0) for k in range(1,K+1)])
LI, LN_ = lagX('ins'), lagX('ne')
def prof(dmu):
    out = {}
    for cd in np.unique(codes):
        m = codes==cd; X = np.column_stack([LI[m], LN_[m]]); b, *_ = np.linalg.lstsq(X, dmu[m], rcond=None); out[cd[:12]] = np.round(b, 3)
    return out
PI_ = np.linalg.lstsq(Pj, rI, rcond=None)[0]; PN_ = np.linalg.lstsq(Pj, rN, rcond=None)[0]
Wo['core.prelude.0.attn.kv.weight'] = KV0.copy(); Wo['core.prelude.0.attn.kv.weight'][288:336] += np.outer(PI_, w_ins) + np.outer(PN_, w_ne)
_, mu1 = run_rows(H100); Wo['core.prelude.0.attn.kv.weight'] = KV0.copy()
print('own-dir x1 extra: lag coefs [ins l1..4 | ne l1..4] per code:'); 
for k, v in prof(mu1-mu0).items(): print(' ', k, v)
df2 = G3[H100[0]].copy(); m = df2.event.isin(['insulin','norepinephrine']); df2.loc[m,'rate'] *= 2; df2.loc[m,'dose'] *= 2
Bx = bundles_of_stay(df2, RU0); Ux = np.array([b['u'] for b in Bx if b['nA']>0]); Xx = lnp(Ux, n1w, n1b)
ins2 = df2[df2.event=='insulin'].rate.values; ne2 = df2[df2.event=='norepinephrine'].rate.values
print('x2 decode ins', np.round(Xx@w_ins,2)[:8], 'true', ins2[:8]); print('x2 decode ne', np.round(Xx@w_ne,3)[:8], 'true', np.round(ne2[:8],3))
```