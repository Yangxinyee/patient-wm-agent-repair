The v3 states show two problems, and together they explain the null dose_edit result.

1. **The edit perturbs bundles it should not touch.** The state already differs, by up to 27 in norm, at bundles before any drug record. The head-1 value change is not confined to the action token. Each added term reads the normalised token with a very large vector (norm about 12,000, needed because the action token stores rate in tiny components). Head 1 gives the slot, event and z tokens only about 0.2% attention, but applied to such a large readout that small weight can produce big spurious injections. A non-zero no-action token readout could add to this.
2. **The intended direction fits poorly.** The direct state change does not match `Jm`@p (cos −0.24), and an injected slot change decays fast (about 40% left after one bundle). So the slot route carries little forward even when it works.

Next I'll pin down the leak. If the cached normalised tokens `L1T` (7 per drug bundle) and attention weights `attd` are what their shapes suggest, I'll compute the three readouts on every token type and weight them by head-1 attention. I'll also check the state change at each stay's very first bundle, which has no history.
- **Leak confirmed:** non-action tokens give readouts in the hundreds or thousands, and their attention-weighted sum matches the spurious changes. Then the readouts must be made orthogonal to those token subspaces, or a different carrier is needed.
- **Leak refuted:** the readouts are near zero off the action token. Then the spurious change has another source and I'll look elsewhere.

```python
print('L1T',L1T.shape,'attd',attd.shape, 'attd sums over last axis', np.round(attd[:3].sum(-1),3).tolist())
print('attd mean over bundles (rows=dim1, cols=tokens):'); print(np.round(attd.mean(0),4))
for j in range(7):
    xj=L1T[:,j]
    r=[xj@BB3[k] for k in ('tI','fN','pI')]
    print(f'token {j}: |x| {np.linalg.norm(xj,axis=1).mean():.2f}  tI mean {r[0].mean():+.2f} sd {r[0].std():.2f} | fN {r[1].mean():+.2f} sd {r[1].std():.2f} | pI {r[2].mean():+.2f} sd {r[2].std():.2f}')
# which token is u? compare with LN1(Uk)
for j in range(7): print(j, 'max|L1T_j - LN1(Uk)| = %.3g'%np.abs(L1T[:,j]-LN1(Uk)).max())
# first bundle of each stay: state change
fb=np.r_[0,np.where(sid[1:]!=sid[:-1])[0]+1]
print('first bundles: hours', np.unique(hr[fb])[:5], ' median |dh| %.3f max %.3f; median |dout| %.4f'%(np.median(nH[fb]),nH[fb].max(),np.median(nO[fb])))
pre=[r for r in range(len(sid)) if not drug[r] and (r in set(fb) or True)]
# bundles before any drug in stay
bef=[]
for s in np.unique(sid):
    w=np.where(sid==s)[0]; dd_=np.where(drug[w])[0]
    lim=w[dd_[0]] if len(dd_) else w[-1]+1
    bef+= [r for r in w if r<lim]
bef=np.array(bef); print('pre-drug bundles n %d: median |dh| %.3f, 90pct %.3f, max %.3f'%(len(bef),np.median(nH[bef]),np.percentile(nH[bef],90),nH[bef].max()))
```