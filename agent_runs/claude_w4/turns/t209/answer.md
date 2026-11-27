`U_of` reproduces every drug bundle's u_k exactly. So I can build exact prelude inputs for the doses the evaluation will apply. Insulin will be scaled ×0 (atoms kept at rate 0, which I assume is how stopping is done), ×0.5, ×1.5, ×2 and ×3 (capped at 11.5, where the encoding saturates). NE will be scaled ×0, ×0.5, ×2 and ×3.

I'll refit the four head-1 readouts (insulin rate and presence, NE rate and presence) on the factual and edited bundles together, using exact attention weights. If errors stay near the factual-only level (rms about 0.02, presence max about 0.03), the carriers will behave correctly under every scored intervention.

```python
def tok_with(mI, mN):
    selr = np.where((iI if mI!=1 else np.ones_like(iI)) & (iN if mN!=1 else np.ones_like(iN)))[0]
    U_ = torch.stack([torch.as_tensor(u_for(rI[i]*mI if iI[i] else 0.0, rN[i]*mN if iN[i] else 0.0) if not ((mI==0 and iI[i]) or (mN==0 and iN[i])) else
                      U_of(([ins_atom(rI[i]*mI)] if iI[i] else []) + ([ne_atom(rN[i]*mN)] if iN[i] else []))).reshape(-1).float() for i in selr])
    T_ = TOK[selr].clone(); T_[:,5,:] = U_
    return selr, T_
print('rate-0 atom test:', np.round(np.asarray(U_of([ins_atom(0.0)])).reshape(-1)[:3],3))
VAR = {}
for mI, mN in [(1,1), (0,1), (0.5,1), (1.5,1), (2,1), (3,1), (1,0), (1,0.5), (1,2), (1,3)]:
    selr, T_ = tok_with(mI, mN)
    rr = prelude2(T_, W_)
    F_ = np.einsum('njt,ntd->njd', rr['att'][:,1,:4,:].numpy(), L1(T_).numpy()).reshape(-1,192)
    VAR[(mI,mN)] = dict(selr=selr, F=F_, tI=np.repeat(np.minimum(rI[selr]*mI, 11.5)*iI[selr], 4), tN=np.repeat(rN[selr]*mN*iN[selr], 4),
                        pI=np.repeat(iI[selr].astype(float),4), pN=np.repeat(iN[selr].astype(float),4))
FF = np.concatenate([VAR[k]['F'] for k in VAR] + [Fn]); 
TT = {nm: np.concatenate([VAR[k][nm] for k in VAR] + [np.zeros(len(Fn))]) for nm in ['tI','tN','pI','pN']}
wv = np.concatenate([np.ones(len(VAR[k]['F'])) for k in VAR] + [10*np.ones(len(Fn))])
sw = np.sqrt(wv); A_ = FF*sw[:,None]; G_ = A_.T@A_
BB = {}
for nm in ['tI','tN','pI','pN']:
    BB[nm] = np.linalg.solve(G_ + 1e-6*np.eye(192), A_.T@(TT[nm]*sw))
    e_ = FF@BB[nm] - TT[nm]
    off = 0; s_ = []
    for k in VAR:
        n_ = len(VAR[k]['F']); ek = e_[off:off+n_]; off += n_
        s_.append('%s:%.3f' % (k, np.abs(ek).max()))
    print(nm, '|b| %.0f rms %.4f' % (np.linalg.norm(BB[nm]), np.sqrt((e_[:-len(Fn)]**2).mean())), 'max by var', ' '.join(s_), 'meas %.3f' % np.abs(e_[-len(Fn):]).max())
```