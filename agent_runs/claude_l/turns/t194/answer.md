Results:

**Expected dose from context is weak.** A linear map from the model's context predicts the insulin rate with CV R² 0.57 (residual sd 0.65) and the norepinephrine rate with R² 0.46. The policy itself is far more predictable from the record (R² 0.82–0.94). This fits two facts: the policy uses glucose thresholds and titrates norepinephrine from the previous rate, and the model's state does not hold the previous rate.

**The error regression shows what the model under-weights.** Glucose errors depend on insulin residuals at lags 1–3: −1.0, −1.4 and −0.5 for finger, and similar for lab. Lags 4–12 show a positive tail of about +0.4: given recent readings, an insulin-caused dip recovers. Arterial MAP errors depend on norepinephrine residuals at +1.85 (lag 1) and +0.7 (lag 2), followed by a negative tail. The expected-dose lags are not zero: +0.56 for insulin, −2.0 for norepinephrine. My linear expected dose is therefore not the one the model has absorbed. A better (nonlinear) expected dose should shrink those coefficients.

**NLL budget.** Glucose errors are large (sd 12.5), so glucose corrections of order 1–2 mg/dL per U/h barely move NLL. MAP errors are sharp (sd 4.4), so norepinephrine corrections must stay close to what the data support.

Next I'll test whether a nonlinear function of context predicts the doses much better. `z_post` has 192 gelu units, so a random-features ridge, which I could write into it directly, is the relevant test. A boosted-tree model gives an upper bound on what context holds.
- **Upper bound near 0.8:** a nonlinear expected dose inside `z_post` is worthwhile.
- **Upper bound stays near 0.6:** the model's state lacks the policy's inputs, and I will tune gains offline against teacher-forced NLL directly.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
mu_c=ctxA.mean(0); U_,S_,Vt_=np.linalg.svd(ctxA[::5]-mu_c,full_matrices=False)
npc=64; Pc_=Vt_[:npc]/ (S_[:npc,None]/np.sqrt(len(ctxA[::5])))   # whitening projection
Zc=(ctxA-mu_c)@Pc_.T
rng=np.random.default_rng(1)
def rf_feats(Z,nf,scale,seed=1):
    r_=np.random.default_rng(seed); Wr=r_.normal(0,scale/np.sqrt(Z.shape[1]),(Z.shape[1],nf)); br=r_.normal(0,1,nf)
    x=Z@Wr+br; return x*0.5*(1+erf(x/np.sqrt(2))), (Wr,br)
for drug,col,hcol in [('ins',2,4),('ne',3,5)]:
    m=meta[:,hcol]==1; yd=meta[m,col]; gd=groups[m]
    for nm_,Xf in [('lin64',Zc[m]),('rf150 s1',rf_feats(Zc[m],150,1.0)[0]),('rf150 s2',rf_feats(Zc[m],150,2.0)[0])]:
        pred=np.zeros(m.sum())
        for trn,tst in GroupKFold(4).split(Xf,yd,gd):
            pred[tst]=Ridge(alpha=1.0).fit(Xf[trn],yd[trn]).predict(Xf[tst])
        print(drug,nm_,'CV R2 %.3f'%(1-((yd-pred)**2).mean()/yd.var()))
    pred=np.zeros(m.sum())
    for trn,tst in GroupKFold(4).split(Zc[m],yd,gd):
        pred[tst]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.1).fit(Zc[m][trn],yd[trn]).predict(Zc[m][tst])
    print(drug,'HGB on 64 PCs CV R2 %.3f'%(1-((yd-pred)**2).mean()/yd.var()))
```