The architecture confirms that the state update uses the z posterior mean, i.e. the first 32 outputs of z_post, and that the value heads average over prior draws, which z_head = 0 already makes inert. So a repurposed z token is deterministic in the evaluation's rollout, and the log-sd outputs can be ignored.

Next I need the injection direction v in token space. I will:
- set the z token to an exact copy of u, then check on one stay that this leaves teacher-forced predictions unchanged (heads 0–2 should just split between two identical tokens);
- at drug bundles of 20 tuning stays, add small perturbations along each of the 192 token axes and measure the change in the predicted means of the four readings at the next bundle, giving a 4×192 response Jacobian per bundle.

I'm looking for a direction that moves finger and lab glucose together, leaves MAP alone, and is consistent across bundles; the analogue for NE should move arterial and cuff MAP. A low cosine between bundles' Jacobians would mean no fixed direction gives a stable gain, and I'd need state-dependent scaling.

```python
import torch, math, numpy as np, time
Wc = {n: torch.tensor(load(n)) for n in NAMES}
codes4 = sorted(MEASC)
try: print('codes4', [(c, CODES[c]) for c in codes4])
except Exception as ex: print('codes4', codes4)
copyz = lambda ctx, e, u, info: u
s0 = list(BUN)[0]
Ra = tf_stay_z(Wc, BUN[s0], None); Rb = tf_stay_z(Wc, BUN[s0], copyz)
print('copy-u check: max |dmean| %.2e, dNLL %.2e' % (max(abs(a[2]-b[2]) for a, b in zip(Ra, Rb)), np.mean([b[3]-a[3] for a, b in zip(Ra, Rb)])))
usd = float(np.std(np.asarray(DA_u)[np.asarray(DA_meta)[:,4]>0], axis=1).mean()); eps_ = 0.05*usd
D_eps = torch.eye(192, dtype=T)*eps_
def jac_tok_stay(W, B_, D):
    times = [b[0] for b in B_]; nd = D.shape[0]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1,4,192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; reg = {}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']] = (times[0], a['num'][0], a['vb'])
    res = []; Dz = torch.cat([torch.zeros(1,192,dtype=T), D], 0); Bt = nd+1
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']] = (t, a['num'][0], a['vb'])
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        if ua and k+1 < len(B_):
            o2, h2, _ = bstep_z(W, out.repeat(Bt,1), h.repeat(Bt,1,1), dt, e[None].repeat(Bt,1), u[None].repeat(Bt,1), Mt.repeat(Bt,1,1), ages, zfn=lambda c_, e_, u_, i_: u_+Dz)
            tn = times[k+1]; views = bview(W, o2, h2, (tn-t)*3600.0, zs=torch.zeros(Bt,1,32,dtype=T)); ms = []
            for c in codes4:
                if c in reg: tl, zl, vl = reg[c]; hs = 1.0; age = (tn-tl)*3600.0
                else: hs, age, zl, vl = 0.0, 0.0, 0.0, 0
                p = bvalue_probs(W, views, c, torch.full((Bt,),hs,dtype=T), torch.full((Bt,),age,dtype=T), torch.full((Bt,),float(zl),dtype=T), torch.full((Bt,),int(vl),dtype=torch.long))
                ms.append((p[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1))
            ms = torch.stack(ms, 1).numpy(); inf = info_of(atoms)
            res.append((t, float(inf['hi']), float(inf['hn']), float(inf['di']), float(inf['dn']), ms[0], (ms[1:]-ms[0])/eps_))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=copyz)
    return res
t0 = time.time(); JT = []
for s in list(BUN)[:20]: JT += [(s,)+r for r in jac_tok_stay(Wc, BUN[s], D_eps)]
print('n drug bundles', len(JT), 'time %.0fs' % (time.time()-t0), 'eps', round(eps_,4))
Jall = np.stack([r[7] for r in JT])  # (n,192,4)
hi = np.array([r[2] for r in JT]); hn = np.array([r[3] for r in JT])
def cos(a, b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for nm_, m, ci in [('ins-only', (hi>0)&(hn==0), (0,1)), ('ne-only', (hn>0)&(hi==0), (2,3)), ('both', (hi>0)&(hn>0), (0,2))]:
    if m.sum() < 3: print(nm_, 'n', m.sum()); continue
    Jm_ = Jall[m].mean(0)
    cons = np.mean([cos(Jall[i][:,ci[0]], Jm_[:,ci[0]]) for i in np.where(m)[0]])
    print(nm_, 'n', m.sum(), '|Jmean| per code', np.round(np.linalg.norm(Jm_, axis=0), 2).tolist(), 'cos(bundle,mean) %.2f' % cons,
          'cos(c%d,c%d) %.2f' % (ci[0], ci[1], cos(Jm_[:,ci[0]], Jm_[:,ci[1]])))
```