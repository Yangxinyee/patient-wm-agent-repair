The factual file has one row per reading (5,177 readings on the 100 setA stays): stay, hour, code, observed value, the tool's predicted mean, and the fine-bin NLL. These are the targets for the head replica.

From the architecture, the view for predicting bundle k is ctx_k + z_head(z), with z drawn from the prior z_prior(ctx_k). Then q = query_norm(view + type_emb), and the value head reads q plus the code embedding and a "last value of this code" register.

The register's 8 numeric inputs and the activation function are not fully documented. My earlier empirical layout gave an error of 2.4 against a noise floor of about 1.2.

So I will test a few register layouts and both activations on about 700 readings from 12 stays:
- the expectation over 64 prior draws is compared with the tool's means;
- the noise floor comes from my own 4-draw averages.

A variant whose error matches the floor, with matching NLL, gives an exact head. Then local factual NLL and rollouts become trustworthy. If every variant stays well above the floor, a head detail is still missing; I would look at type_emb indices or the last_status term next.

```python
VBC = VOCAB['value_bin_centres']; print(type(VBC), (list(VBC.keys()) if isinstance(VBC, dict) else len(VBC)))
CENT = {c: np.array(VBC[c], float) for c in VBC} if isinstance(VBC, dict) else None
rng = np.random.default_rng(0)
def head_value(ctx, code, last, opt, nz=64, zs=None):
    pr = mlp2n(ctx, 'z_prior'); mu, ls = pr[:32], pr[32:]
    if zs is None: zs = rng.standard_normal((nz,32))
    Z = mu + np.exp(ls)*zs
    V = ctx[None] + linn(Z, 'z_head')
    q = LNn(V + Wo['type_emb.weight'][opt['ty']], 'query_norm')
    has, agemin, zl, vb = last
    lv = {'A':[has, np.log1p(agemin*opt['au'])/10, zl, has, has, 0,0,0],
          'B':[has, np.log1p(agemin*opt['au'])/10, zl, 0, 0,0,0,0],
          'C':[has, np.log1p(agemin*opt['au'])/10, zl, has, 0,0,0,0]}[opt['ln']]
    wc = q + Wo['embed.code.weight'][CID[code]] + linn(np.array(lv,float),'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
    if opt.get('ls') is not None: wc = wc + Wo['last_status.weight'][opt['ls']]
    a = linn(wc, 'attr_heads.value_bin.0'); a = gelu(a) if opt['act']=='gelu' else np.maximum(a,0)
    lg = linn(a, 'attr_heads.value_bin.2') + Wo['embed_transition.value_bin.weight'][CID[code]*17 + vb]
    lg = lg - lg.max(-1, keepdims=True); P = np.exp(lg); P /= P.sum(-1, keepdims=True)
    return P   # nz x 17
def eval_head(opt, stays, nmax=700):
    out=[]
    for s in stays:
        B = bundles_of_stay(tuA[tuA.stay_id==s], RU0); rows = np.where(sid3==s)[0]
        fs = F2[F2.stay_id==s]; tk = np.array([b['t'] for b in B])
        for r in fs.itertuples():
            k = int(np.argmin(np.abs(tk - r.hour)))
            if k==0 or abs(tk[k]-r.hour)>1e-6: continue
            dt = (tk[k]-tk[k-1])*3600; ctx = ctx_of(X3[rows[k-1]], dt)
            prev = tu[(tu.stay_id==s)&(tu.event==r.event)&(tu.hour<r.hour-1e-9)]
            if len(prev): lvv = prev.value.iloc[-1]; zl, vb = zb(r.event,'value',lvv); last=(1,(r.hour-prev.hour.iloc[-1])*60, zl, vb)
            else: last=(0,0,0,0)
            P = head_value(ctx, r.event, last, opt)
            cen = CENT[r.event] if CENT else np.array(VBC, float)
            m_all = P[:,1:]@cen
            m_exp = m_all.mean()
            m4 = np.array([m_all[i:i+4].mean() for i in range(0,64,4)])
            _, ob = zb(r.event,'value',r.observed)
            nll = -np.log(P[:, ob].mean())
            out.append((r.predicted_mean, m_exp, m4.std(), r.nll, nll))
            if len(out)>=nmax: return np.array(out)
    return np.array(out)
for opt in [dict(ty=1,ln='A',au=1,act='gelu'), dict(ty=1,ln='A',au=1,act='relu'), dict(ty=1,ln='B',au=1,act='gelu'),
            dict(ty=1,ln='C',au=1,act='gelu'), dict(ty=1,ln='A',au=60,act='gelu'), dict(ty=1,ln='A',au=1,act='gelu',ls=0)]:
    o = eval_head(opt, setA_ids[:12], nmax=700)
    print(opt, 'n',len(o),'MAE mean %.3f  floor %.3f | nll tool %.4f mine %.4f | mean|dnll| %.4f' % (np.abs(o[:,0]-o[:,1]).mean(), o[:,2].mean(), o[:,3].mean(), o[:,4].mean(), np.abs(o[:,3]-o[:,4]).mean()))
```