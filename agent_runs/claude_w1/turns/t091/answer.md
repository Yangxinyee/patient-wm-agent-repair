The head replica is close but not exact. The best variants have a mean-prediction error of about 1.86–2.0 against a sampling floor of about 1.3, and an average per-reading NLL gap of about 0.19. Some systematic difference remains.

Of the options tested:
- the register age works in minutes (seconds is clearly worse);
- the register layout matters little between A and C, but the version without the extra "has" flags is much worse;
- GELU and ReLU are close.

The most likely remaining cause is the view's time-gap term. The architecture text writes `gap(log dt)` for the view but `gap(log dt / 10)` for the context. I assumed the two were identical.

Next I test:
- the view's gap term without the /10;
- the register age in hours;
- a couple of alternative type indices.

A variant that brings the error down to the floor, with a NLL gap near pure sampling noise, would be adopted. If nothing changes, the discrepancy lies elsewhere, perhaps in how the 4 prior draws are combined.

```python
def head_value2(prev, dt, code, last, opt, zs):
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    base = prev[:192] + LNn(h.mean(0),'core.out_norm')
    ctx = base + mlp2n(np.array([np.log(dt)/10]),'gap')
    pr = mlp2n(ctx, 'z_prior'); Z = pr[:32] + np.exp(pr[32:])*zs
    gv = mlp2n(np.array([np.log(dt)/opt['gdiv']]),'gap')
    V = base[None] + gv + linn(Z, 'z_head')
    q = LNn(V + Wo['type_emb.weight'][opt['ty']], 'query_norm')
    has, agemin, zl, vb = last
    lv = [has, np.log1p(agemin*opt['au'])/10, zl, has, has, 0,0,0]
    wc = q + Wo['embed.code.weight'][CID[code]] + linn(np.array(lv,float),'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
    a = linn(wc, 'attr_heads.value_bin.0'); a = gelu(a) if opt['act']=='gelu' else np.maximum(a,0)
    lg = linn(a, 'attr_heads.value_bin.2') + Wo['embed_transition.value_bin.weight'][CID[code]*17 + vb]
    lg = lg - lg.max(-1, keepdims=True); P = np.exp(lg); return P/P.sum(-1, keepdims=True)
# precompute reading contexts once
READ = []
for s in setA_ids[:12]:
    B = bundles_of_stay(tuA[tuA.stay_id==s], RU0); rows = np.where(sid3==s)[0]; tk = np.array([b['t'] for b in B])
    for r in F2[F2.stay_id==s].itertuples():
        k = int(np.argmin(np.abs(tk - r.hour)))
        if k==0 or abs(tk[k]-r.hour)>1e-6: continue
        prev = tu[(tu.stay_id==s)&(tu.event==r.event)&(tu.hour<r.hour-1e-9)]
        if len(prev): zl, vb = zb(r.event,'value',prev.value.iloc[-1]); last=(1,(r.hour-prev.hour.iloc[-1])*60, zl, vb)
        else: last=(0,0,0,0)
        READ.append(dict(prev=X3[rows[k-1]], dt=(tk[k]-tk[k-1])*3600, code=r.event, last=last, pm=r.predicted_mean, nll=r.nll, ob=zb(r.event,'value',r.observed)[1]))
READ = READ[:400]
ZS = np.random.default_rng(1).standard_normal((64,32))
def eval2(opt):
    o=[]
    for d in READ:
        P = head_value2(d['prev'], d['dt'], d['code'], d['last'], opt, ZS)
        m_all = P[:,1:]@CENT[d['code']]
        o.append((d['pm'], m_all.mean(), np.array([m_all[i:i+4].mean() for i in range(0,64,4)]).std(), d['nll'], -np.log(P[:,d['ob']].mean())))
    o=np.array(o); return np.abs(o[:,0]-o[:,1]).mean(), o[:,2].mean(), o[:,3].mean()-o[:,4].mean(), np.abs(o[:,3]-o[:,4]).mean()
for opt in [dict(gdiv=10,ty=1,au=1,act='gelu'), dict(gdiv=1,ty=1,au=1,act='gelu'), dict(gdiv=1,ty=1,au=1,act='relu'),
            dict(gdiv=10,ty=1,au=1/60,act='gelu'), dict(gdiv=10,ty=0,au=1,act='gelu'), dict(gdiv=10,ty=2,au=1,act='gelu')]:
    print(opt, 'MAE %.3f floor %.3f  dNLL(tool-mine) %.4f  |dNLL| %.4f' % eval2(opt))
```