The gate is clean. The regularised LDA direction mostly reads the code embedding. Standardised so glucose ≈ +1 and MAP ≈ −1:
- glucose readings have minimum 0.88 and sd 0.014; MAP readings have maximum −0.94 and sd 0.014 (α = 0.01);
- with α = 0.1 the margins are still wide: glucose ≥ 0.76, MAP ≤ −0.81;
- drug codes sit near 0.16 and −0.28, and visit codes near 0.

With gain about 16 and threshold 0.5, glucose-gated units sit deep in GELU's linear range, about +8. MAP and other atoms sit far in the dead range, below −5, and the reverse holds for the MAP gate.

`eps_ins` has sd 0.56 units/h and `eps_ne` has sd 0.084 mcg/kg/min. Both are dose minus the mean dose for the glucose band and previous on/off status (`mean_tab`), so they are the "dose minus readings-implied dose" residuals I want the probe to read.

Next I'll measure two things:
- **Missing dose effects:** how much of the model's real factual error, observed minus the tool's predicted mean, is explained by these residuals at lags 1–4. This gives β, the dose effects the model fails to use given its own information.
- **Probe recoverability:** how well each lag's residual can be read from with_code by ridge regression, with cross-validation grouped by stay.

β near (−1.5, −1.4, 0, …) for insulin, matching my earlier notes, and a cross-validated probe R² of about 0.3 or more would confirm the plan. A near-zero R² would mean the dose information must be read elsewhere, for example from the posterior z.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
si_=np.array([sid2i[s] for s in M4.stay_id.values]); hr_r=M4.hour.values.astype(int)
def lagres(E,l):
    t=hr_r-l; ok=(t>=0)&(t<=24); out=np.zeros(len(t)); out[ok]=E[si_[ok],t[ok]]; return out
RI_=np.stack([lagres(eps_ins,l) for l in range(1,5)],1); RN_=np.stack([lagres(eps_ne,l) for l in range(1,5)],1)
err_=M4.observed.values-pmall
for nm_,msk_,R_ in [('glu',glu,RI_),('map',mapm,RN_)]:
    X_=np.column_stack([np.ones(msk_.sum()),R_[msk_]]); bb,*_=np.linalg.lstsq(X_,err_[msk_],rcond=None)
    rs=err_[msk_]-X_@bb; se=np.sqrt(np.diag(np.linalg.inv(X_.T@X_))*rs.var())
    print(nm_,'err ~ resid lags1-4: beta',np.round(bb[1:],2),'se',np.round(se[1:],2))
    grp_s=M4.stay_id.values[msk_]
    for l in range(3):
        y_=R_[msk_][:,l]
        for al in [10.0,100.0]:
            pr=np.zeros_like(y_)
            for tri,tei in GroupKFold(5).split(WCA[msk_],y_,grp_s):
                m=Ridge(alpha=al).fit(WCA[msk_][tri],y_[tri]); pr[tei]=m.predict(WCA[msk_][tei])
            print(f'  probe lag{l+1} alpha{al}: CV R2 %.3f'%(1-((y_-pr)**2).mean()/y_.var()))
```