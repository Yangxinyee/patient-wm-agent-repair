This is the clearest result so far, from 400 stays.

**Glucose.** The model's factual residuals load on the insulin surprise at lag 1 (−1.56 ± 0.33), lag 2 (−1.95 ± 0.33) and lag 3 (−0.47). They do not load on the protocol-expected dose (about 0). So the model absorbed the expected dose through the readings and mostly ignores the random part. True lag-1 −1.99 minus the missing −1.56 leaves about −0.43 that the model has, which matches its measured rollout response of −0.45.

**MAP.** The residuals load on +2.9·NE₁ − 1.3·NE₂ − 1.8·NE₃, which sums to about zero. The model is missing the effect of the latest NE change relative to what the MAP history already implies.

Both patterns fit one rule. The correction should be the true effect times (dose − the dose predictable from the readings history). That rule improves factual fit and, under a sustained edit, adds roughly θ₁·Δdose every hour.

The next question is whether these targets can be read linearly from the value head's input. I'll fit cross-validated ridge decoders from `with_code`:
- for glucose, the lag-1 and lag-2 surprises and the residual-weighted combination;
- for MAP, the residual-weighted NE combination and NE₁ alone.

A cross-validated R² of about 0.4 or more for the lag-1 signals would make a head-level correction workable. A low R² for lag 2 would confirm that the lag-2 effect must come from overshoot and persistence.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
WC0 = Q_all + HI['reg']   # with_code (mean z)
RD2['row_i']=np.arange(len(RD2))
def dec_r2(mask, y, alphas=(1,3,10,30,100)):
    Xw = WC0[mask]; mu=Xw.mean(0); sd=Xw.std(0)+1e-6; Xz=(Xw-mu)/sd
    gk = KFold(5, shuffle=True, random_state=0)
    rc = RidgeCV(alphas=alphas).fit(Xz,y); pr = cross_val_predict(RidgeCV(alphas=alphas), Xz, y, cv=gk)
    return 1-((y-pr)**2).sum()/((y-y.mean())**2).sum(), rc.alpha_
gmask = RD2.event.str.startswith('glucose').values & RD2[['eps_ins_1','eps_ins_2','eps_ins_3']].notna().all(1).values
mmask = RD2.event.str.startswith('map').values & RD2[['ne_1','ne_2','ne_3']].notna().all(1).values
d = RD2[gmask]
for nm,y in [('eps1',d.eps_ins_1.values),('eps2',d.eps_ins_2.values),('ins1',d.ins_1.values),('yins',-1.56*d.eps_ins_1.values-1.95*d.eps_ins_2.values-0.47*d.eps_ins_3.values)]:
    print(nm, [round(v,3) for v in dec_r2(gmask,y)], end=' | ')
print()
d = RD2[mmask]
for nm,y in [('yne',2.86*d.ne_1.values-1.25*d.ne_2.values-1.83*d.ne_3.values),('ne1',d.ne_1.values),('dne',d.ne_1.values-d.ne_2.values)]:
    print(nm, [round(v,3) for v in dec_r2(mmask,y)], end=' | ')
```