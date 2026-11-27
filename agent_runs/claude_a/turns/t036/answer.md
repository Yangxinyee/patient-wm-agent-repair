Two things confirmed. First, the states layout: out equals out_norm applied to the mean of the 4 slots, with error 5e-7. Second, I already have the Forward-2 factual results matched to state rows (`fr` has row_self, row_prev, dt), so the prediction heads can now be checked one reading at a time.

From the vocabulary, the insulin and norepinephrine dose and rate inputs carry the same z-score; for norepinephrine this is because dose = 4.8 × rate. So the dose enters each drug atom as (numeric columns 2 + 4) × z plus two bin embeddings. Doubling a dose pushes many insulin rates into the top bin 16, with z up to about 8.

Next I will rebuild the value head on CPU from the stored states:
- context from out, the decayed h and the gap MLP;
- the z prior mean in place of the 4 random draws;
- the query norm, the "last" register and the value MLP plus the transition term.

I will compare the result with the model's reported predicted means while searching the unknowns: the activation, the layout of the 8 `last_numeric` features, the age units, the gap scaling, the status row and the type index. If one variant reproduces the reported means closely, with only the small noise from the z draws left, the heads are pinned down. If none does, the type index or the register layout is wrong and I will widen the search.

```python
import torch, numpy as np, pandas as pd, itertools
F = torch.nn.functional
P = {n: load(n) for n in NAMES}
print(fr.head(4).to_string()); print('dt', fr.dt.describe().to_dict())
fa_ = P['core.flow.a']; print('timescales s', (1/torch.exp(fa_)).min().item(), (1/torch.exp(fa_)).max().item())
print('last_numeric col norms', P['last_numeric.weight'].norm(dim=0).numpy().round(3))
print('bins row0 norms', P['embed.bins.value.weight'][0].norm().item(), P['embed.bins.rate.weight'][0].norm().item())
X = torch.tensor(S['x']); OUT = X[:, :192]; HS = X[:, 192:960].reshape(-1,4,192); ZP = X[:, 960:]
acts = {'gelu': F.gelu, 'relu': F.relu, 'silu': F.silu}
def lin(x, n): return F.linear(x, P[n+'.weight'], P.get(n+'.bias'))
def mlp(x, n, act): return lin(act(lin(x, n+'.0')), n+'.2')
def ln(x, n): return F.layer_norm(x, (192,), P[n+'.weight'], P[n+'.bias'], eps=1e-5)
num = {(e['code'], e['attribute']): e for e in VOCAB['numeric']}
cid = {c['name']: c['id'] for c in VOCAB['codes']}
cent = {k: torch.tensor(v, dtype=torch.float32) for k, v in VOCAB['value_bin_centres'].items()}
def zsc(code, attr, x): e = num[(code, attr)]; return np.clip((x - e['median'])/e['iqr'], -10, 10)
def fbin(code, attr, x): e = num[(code, attr)]; return 1 + np.searchsorted(np.array(e['fine_bin_edges']), x, side='left')
rec2 = tr[tr.stay_id.isin(fr.stay_id.unique()) & tr.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].sort_values(['stay_id','event','hour']).copy()
rec2['lastv'] = rec2.groupby(['stay_id','event']).value.shift(1); rec2['lasth'] = rec2.groupby(['stay_id','event']).hour.shift(1)
R = fr.merge(rec2[['stay_id','hour','event','lastv','lasth']], on=['stay_id','hour','event'], how='left')
assert len(R) == len(fr)
has = (~R.lastv.isna()).values.astype(np.float32)
age_h = np.where(has > 0, R.hour.values - R.lasth.values, 0.0)
zv = np.zeros(len(R), np.float32); vb = np.zeros(len(R), np.int64)
for ev in R.event.unique():
    m = (R.event == ev).values & (has > 0)
    zv[m] = zsc(ev, 'value', R.lastv.values[m]); vb[m] = fbin(ev, 'value', R.lastv.values[m])
codes_ = torch.tensor(R.event.map(cid).values)
kprev = torch.tensor(R.row_prev.values.astype(np.int64))
dtv = torch.tensor(R.dt.values, dtype=torch.float32)
obs_pm = torch.tensor(R.predicted_mean.values, dtype=torch.float32)
centM = torch.zeros(13, 16)
for k_, v_ in cent.items(): centM[cid[k_]] = v_
def head_mean(act, dt_s, gapdiv, layout, age_s, status, typ_off, renorm):
    A = acts[act]
    hdec = HS[kprev] * torch.exp(-dt_s[:, None, None] * torch.exp(P['core.flow.a']))
    g = mlp((torch.log(dt_s)/gapdiv)[:, None], 'gap', A)
    ctx = OUT[kprev] + ln(hdec.mean(1), 'core.out_norm') + g
    pr = mlp(ctx, 'z_prior', A); z = pr[:, :32]
    view = ctx + lin(z, 'z_head')
    q = ln(view + P['type_emb.weight'][codes_ + typ_off], 'query_norm')
    ag = torch.tensor(np.log1p(age_h*(3600 if age_s else 1))/10, dtype=torch.float32)
    hz = torch.tensor(has); zz = torch.tensor(zv); zero = torch.zeros_like(hz)
    if layout == 'L1': f = torch.stack([hz, ag, zz, zero, zero, zero, zero, zero], 1)
    elif layout == 'L2': f = torch.stack([hz, ag, zz, hz, zero, zero, zero, zero], 1)
    elif layout == 'L4': f = torch.stack([zz, hz, zero, zero, zero, zero, hz, ag], 1)
    else: f = torch.stack([hz, ag, zz, zero, hz, zero, zero, zero], 1)
    wc = q + P['embed.code.weight'][codes_] + lin(f, 'last_numeric') + P['embed.bins.value.weight'][torch.tensor(vb)] + P['embed.bins.rate.weight'][0]
    if status is not None: wc = wc + P['last_status.weight'][status]
    lg = mlp(wc, 'attr_heads.value_bin', A) + P['embed_transition.value_bin.weight'][codes_*17 + torch.tensor(vb)]
    p = torch.softmax(lg, -1)
    pb = p[:, 1:]
    if renorm: pb = pb / pb.sum(1, keepdim=True)
    return (pb * centM[codes_]).sum(1), p[:, 0]
res = []
dts_opts = {'s': dtv if R.dt.max() > 100 else dtv*3600}
for act, gapdiv, layout, age_s, status, renorm in itertools.product(['gelu','relu','silu'], [10.0, 1.0], ['L1','L2','L4','L5'], [True, False], [None, 0, 1], [False]):
    m_, p0 = head_mean(act, dts_opts['s'], gapdiv, layout, age_s, status, 0, renorm)
    err = (m_ - obs_pm); gm = (R.event=='glucose_finger').values; mm = (R.event=='map_arterial').values
    res.append((err[gm].pow(2).mean().sqrt().item(), err[mm].pow(2).mean().sqrt().item(), act, gapdiv, layout, age_s, status, p0.mean().item()))
res.sort(key=lambda r: r[0] + r[1])
for r in res[:8]: print(r)
```