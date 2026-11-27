The norepinephrine result is stronger still: the factual MAP residuals rise with the NE surprise at every lag. Arterial slopes are 3.4, 2.7, 3.2 (se about 0.75) and cuff slopes 4.4, 2.7, 1.4, against true effects of 3.3, 2.7, 2.0. So the model's factual predictions ignore the NE effect, and at lag 3 they lean slightly the wrong way, consistent with the confounded learning. Both drugs therefore need the missing response added, centred on the surprise, which should also help factual fit.

Next I fit linear decoders from with_code to the surprise at lags 1 to 3:

- insulin surprise on glucose readings;
- NE surprise on MAP readings.

I use ridge with stay-grouped cross-validation. I report the cross-validated R² and the response matrix of each decoded lag to the true surprises. That matrix tells me how much each decoder under-responds, so I can rescale it to give the full counterfactual response.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
def cvdec(rows, Y, alphas=(3,30,300,3000)):
    X = WC0[rows]; mu = X.mean(0); sd = X.std(0)+1e-6; Xs = (X-mu)/sd
    grp = RD2['stay_id'].values[rows]
    best=None
    for a in alphas:
        Yh = np.zeros_like(Y)
        for trn, tst in GroupKFold(5).split(Xs, Y, grp):
            Yh[tst] = Ridge(alpha=a).fit(Xs[trn], Y[trn]).predict(Xs[tst])
        r2 = 1-((Y-Yh)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)
        if best is None or r2.mean()>best[1].mean(): best=(a,r2,Yh)
    a,r2,Yh = best
    Xe = np.column_stack([np.ones(len(Y)),Y])
    M = np.linalg.lstsq(Xe, Yh, rcond=None)[0][1:].T   # M[k,j]: dec k on true j
    return a, np.round(r2,2), np.round(M,2), mu, sd
rg = np.where(isg)[0]; rm = np.where(ism)[0]
Yg = RD2.loc[rg, ['eps_ins_1','eps_ins_2','eps_ins_3']].fillna(0).values
Ym = RD2.loc[rm, ['eps_ne_1','eps_ne_2','eps_ne_3']].fillna(0).values
outg = cvdec(rg, Yg); outm = cvdec(rm, Ym)
print('G', outg[0], outg[1]); print(outg[2])
print('M', outm[0], outm[1]); print(outm[2])
```