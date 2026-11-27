Run 270 established the design:
- The value head's view is `base + z_head(z)`, where `z = z_prior(base)` and `base` is the same context vector the core uses.
- The existing Jacobians from the z token to the predicted means are tiny (median 0.03–0.11 per unit norm). A z-token write alone cannot produce shifts of several units.
- Insulin rates above 5.75 are only 0.5% of records, so the u saturation mainly affects the ×3 edits.

Design:
1. **Write:** at drug bundles, set the z token to a copy of u plus `s·v_j·ρ_j`, where `ρ_j = dose_j − m_j(ctx)` is gated to bundles where drug j is present.
2. **Read:** at prediction time, `z_prior` decodes ρ from base, with σ→0 so it is deterministic.
3. **Amplify:** `z_head` maps the decoded ρ onto a view direction that shifts glucose (code 8) or MAP (code 9) with a set gain.

This run tests three components:
- the u-copy basis centered on the reading-bundle u,
- linear presence and dose decoders from u (insulin capped at 11.5),
- random-feature models of the expected dose `m_j(ctx)`.

Max decoder error ≲0.01 on presence and ≲0.1 on dose, with CV R² ≳0.7 for `m_j`, would confirm the write side is buildable.

```python
from sklearn.linear_model import Ridge
from scipy.special import erf
gnp=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))
ue_=u_empty.numpy()
Uf2=np.vstack([Ug,DAu[::4],ue_[None]]); _,_,Vt2=np.linalg.svd(Uf2-ue_,full_matrices=False); Pk=Vt2[:30].T
rec=lambda X: np.abs((X-ue_)-((X-ue_)@Pk)@Pk.T).max(1)
print('copy maxerr grid %.3f data %.3f empty %.1e'%(rec(Ug).max(),rec(DAu).max(),rec(ue_[None]).max()))
Xu=np.vstack([Ug,DAu[::3],ue_[None]])
Yu=np.vstack([np.c_[np.minimum(Lg[:,0],11.5),Lg[:,1:]], DAm[::3,2:6], [[0,0,0,0]]])
for nm,X in [('full',Xu-ue_),('pc30',(Xu-ue_)@Pk)]:
    R=Ridge(alpha=1e-6).fit(X,Yu); E=np.abs(R.predict(X)-Yu)
    print(nm,'maxerr [d8,d9,p8,p9]',E.max(0).round(3),'p99',np.percentile(E,99,0).round(3))
cm=DAc.mean(0); cs=DAc.std(0)+1e-6; Zc=(DAc-cm)/cs
_,svc,Vtc=np.linalg.svd(Zc[::5],full_matrices=False); Pc=Vtc[:40].T/(svc[:40]/np.sqrt(len(Zc[::5]))); Zw=Zc@Pc
rng=np.random.default_rng(0); A_=rng.normal(size=(40,75))/np.sqrt(40); b_=rng.uniform(-2,2,75); Fm=gnp(Zw@A_+b_)
grp=DAm[:,0].astype(int)%5
for j,(col,pc) in enumerate([(2,4),(3,5)]):
    msk=DAm[:,pc]>0; y=DAm[msk,col]
    for nm,X in [('lin',Zw[msk]),('gelu',Fm[msk]),('both',np.c_[Fm[msk],Zw[msk]])]:
        pr=np.zeros_like(y)
        for f in range(5):
            tr=grp[msk]!=f; pr[~tr]=Ridge(alpha=1.0).fit(X[tr],y[tr]).predict(X[~tr])
        print(['m8','m9'][j],nm,'CV R2 %.3f resid sd %.3f'%(1-np.var(y-pr)/np.var(y),np.std(y-pr)),end=' | ')
    print()
```