The head's inputs are now known: codes 5–10 are map_cuff, glucose_finger, map_arterial, insulin, norepinephrine and glucose_lab. Per-code median, IQR and 15 fine-bin edges are in VOCAB, and the transition table has 13×17 rows.

I'll rebuild the value head locally and fit its two unknowns to the 100 fwd_3 stays. The unknowns are the `type_emb` index for readings and the layout of the 8 `last_numeric` inputs. The first layer of the value MLP is linear in `with_code`, so each candidate layout costs one GELU and one small matmul. That makes an exhaustive search over type × status × column placement cheap. For speed, z is taken as the prior mean for now. A near-exact match (RMSE of a few tenths of a mg/dL against the 4-sample fwd_3 means) would confirm the head. A floor of several mg/dL would mean I've misread something (for example the flow or the gap input).

```python
import pandas as pd, itertools, time
f3 = pd.read_csv('/work/fwd_3/factual_original.csv')
NUMV = {d['code']: d for d in VOCAB['numeric'] if d['attribute']=='value'}
CID = {c['name']: c['id'] for c in VOCAB['codes']}
VCODES = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
CEN = {c: torch.tensor(VOCAB['value_bin_centres'][c], dtype=torch.float64) for c in VCODES}
def zsv(code, x): d=NUMV[code]; return float(np.clip((x-d['median'])/d['iqr'], -10, 10))
def fbv(code, x): return 1 + int(np.searchsorted(np.array(NUMV[code]['fine_bin_edges']), x, side='left'))
gelu = torch.nn.functional.gelu
def lin_t(x, n): return x @ W[n+'.weight'].T + W[n+'.bias']
fa_ = torch.exp(W['core.flow.a'])
tgt = {(int(r.stay_id), int(round(r.hour)), r.event): r.predicted_mean for r in f3.itertuples()}
H = []  # rows
t0_=time.time()
for sid in bundles_B.keys():
    R = run_tf(sid); last = {}
    for k, r in enumerate(R):
        if k > 0:
            pv = R[k-1]
            hfl = pv['y'] * torch.exp(-r['dt']*fa_)
            g_ = W['gap.2.weight'] @ gelu(W['gap.0.weight'][:,0]*(math.log(r['dt'])/10) + W['gap.0.bias']) + W['gap.2.bias']
            v0 = pv['out'] + LNt(hfl.mean(0)[None], 'core.out_norm')[0] + g_
            zp_ = lin_t(gelu(lin_t(v0, 'z_prior.0')), 'z_prior.2')
            view = v0 + lin_t(zp_[:32], 'z_head')
            for a in r['ats']:
                c = a['code']
                if c in VCODES and a['kind']==0 and a.get('value') is not None:
                    key = (int(sid), int(round(r['t'])), c)
                    if key in tgt:
                        L = last.get(c)
                        H.append(dict(view=view, cid=CID[c], code=c, has=L is not None,
                                      lvb=fbv(c, L[0]) if L else 0, age=math.log1p((r['t']-L[1])*3600)/10 if L else 0.0,
                                      zv=zsv(c, L[0]) if L else 0.0, target=tgt[key], obs=a['value']))
        for a in r['ats']:
            if a['code'] in VCODES and a['kind']==0 and a.get('value') is not None:
                last[a['code']] = (a['value'], r['t'])
print("rows:", len(H), "of fwd_3", len(f3), "time %.1fs" % (time.time()-t0_))
print("has_last frac:", np.mean([h['has'] for h in H]).round(3), "| codes:", pd.Series([h['code'] for h in H]).value_counts().to_dict())
print("gapf check:", float((gapf(3600.0) - (W['gap.2.weight'] @ gelu(W['gap.0.weight'][:,0]*(math.log(3600.0)/10) + W['gap.0.bias']) + W['gap.2.bias'])).abs().max()) if callable(gapf) else 'n/a')
```