Insulin is given only at hours with a glucose reading, and the reading used is the finger stick, or the lab value if there is no finger stick:
- **Finger reading present:** tier 0 never gets insulin, and tiers 1–4 get it with probability 0.99–1.0, at mean rates 1.05, 2.07, 3.12 and 3.84.
- **Lab only:** the same pattern.
- **No glucose reading:** never.

So the factual expected rate is a pure step function of that reading (finger tier 1 expected ≈ 0.986 × 1.048 ≈ 1.03). The two-step memory write is consistent with this:
- at the reading bundle, write −rbar(tier of the policy's reading);
- five minutes later, at the infusion bundle, write +rate.

That leaves exactly the randomised deviation in factual data.

**Next step:** fit the write decoder, a linear map from the slot-mean E to that target. It will be a single row of `b_in`, applied at every bundle. Every other bundle type (MAP-only, ends, NE, admission) must decode to 0. I will fit a ridge regression on 240 stays and test it on 60 held-out stays, with R² and residual spread by bundle type.

**What would confirm or refute it:**
- Confirmed: held-out R² ≳ 0.95, with small residuals on the zero-target bundles.
- Refuted: poor decodability. Accumulated write noise would then hurt the factual fit, and I would need a cleaner write path, such as through the prelude's u token.

```python
at_ = dsA['at']; nb = nbA
print("insulin kinds:", at_[at_.event=='insulin'].kind.value_counts().to_dict())
rb_f = {0:0.0,1:0.986*1.048,2:2.069,3:3.116,4:3.835}; rb_l = {0:0.0,1:0.956*1.048,2:2.005,3:2.783,4:3.077}
def tier5(g): return int(np.searchsorted([150,180,220,250], g, side='right'))
yI = np.zeros(nb); btype = np.array(['other']*nb, dtype=object)
gfa = at_[at_.event=='glucose_finger'].groupby('row').value.first()
gla = at_[at_.event=='glucose_lab'].groupby('row').value.first()
for rw in set(gfa.index)|set(gla.index):
    if rw in gfa.index: yI[rw] = -rb_f[tier5(gfa[rw])]
    else: yI[rw] = -rb_l[tier5(gla[rw])]
    btype[rw] = 'gread'
ia = at_[(at_.event=='insulin')&(at_.kind==0)&at_.rate.notna()].groupby('row').rate.sum()
for rw,v in ia.items(): yI[rw] += v; btype[rw] = 'ins'
Eb = EA.mean(1).numpy().astype(np.float64)
stay_of = dsA['bd']['stay_id'].values
sts = np.array(stA); tr_m = np.isin(stay_of, sts[:240]); te_m = ~tr_m
from sklearn.linear_model import Ridge
for alpha in [1e-3, 1e-1, 3]:
    rg = Ridge(alpha=alpha, fit_intercept=False).fit(Eb[tr_m], yI[tr_m]); pr_ = rg.predict(Eb)
    res = yI - pr_
    r2 = 1 - ((res[te_m])**2).sum()/((yI[te_m]-yI[te_m].mean())**2).sum()
    msg = " ".join(f"{t}:sd{res[te_m&(btype==t)].std():.3f}/bias{res[te_m&(btype==t)].mean():+.3f}" for t in ['gread','ins','other'])
    print(f"alpha {alpha}: test R2 {r2:.4f} | {msg}")
print("counts:", {t:int((btype==t).sum()) for t in ['gread','ins','other']}, " yI sd by type:", {t:round(float(yI[btype==t].std()),3) for t in ['gread','ins']})
```