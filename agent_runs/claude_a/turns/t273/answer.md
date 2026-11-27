Now I can see the writer. `build_ins2` repurposes two output dims (jU, jR) of the dose encoder so that their difference ≈ 2·kap·gc × insulin rate. It reads the rate through a linear decoder (`dvec`, `d0`) fitted on the encoder's other dims. `keep` must be those other 189 dims, i.e. everything except the three reserved ones. So `keep` is easy to recompute, and I can rebuild on the cheap dims after refitting the decoder.

Plan for this run, using jU=79, jR=99, jI=50 (all cheap on both test sets):
1. Get per-bundle insulin rates with `bundle_rates`.
2. Refit the rate decoder by ridge regression of the rate on the original dose-encoder output, excluding those dims, over all dose bundles (norepinephrine-only bundles have target 0).
3. Build with a longer half-life (2.5 h, closer to the truth kernel) and a smaller write gain, so the memory stays modest.
4. Measure the factual cost on sel2 and dsE.
5. Check that memory dim 50 tracks an exponentially weighted sum of past insulin rates.

Success means a decoder R² near 0.99, cost ≤0.001 on both sets, and a high correlation between the memory and the weighted dose history.

```python
br=None
for nm_ in ['ins','insulin']:
    try: br=np.asarray(bundle_rates(dsE,nm_)); drugname=nm_; break
    except Exception as ex: err=str(ex)[:80]
print('bundle_rates', drugname if br is not None else err, None if br is None else (br.shape, float((br>0).mean())))
u=np.asarray(dsE['u']); isdrug=np.abs(u).sum(1)>0
Jn=[79,99,50]; keepN=np.array([k for k in range(192) if k not in Jn])
X=u[isdrug][:,keepN]; y=br[isdrug]
from sklearn.linear_model import Ridge
rg=Ridge(alpha=1e-3).fit(X,y); pr=rg.predict(X)
print('decoder R2 %.4f resid sd %.3f  (ins bundles sd %.3f) n=%d'%(1-np.var(y-pr)/np.var(y), np.std(y-pr), np.std(y[y>0]), len(y)))
dvecN=rg.coef_.astype(np.float32); d0N=float(rg.intercept_)
PA=build_ins2(PO, dvecN, d0N, keepN, jU=79, jR=99, jI=50, half_life_h=2.5, cb=0.4)
print('cost sel2 %+.5f  dsE %+.5f'%(dcost(PA,ds_s2,a_s2), dcost(PA,dsE,a_E)))
```