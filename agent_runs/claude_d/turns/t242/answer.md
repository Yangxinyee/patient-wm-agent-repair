The output mostly shows that the helper signatures still exist (build_ins3, eval_E, head_mean2, aug_wc, cvdec_aug2, ridge_gate). The reading table has NE lag columns ne_1..3, eps_ne_1..3 and D_ne_1..3. The long print came from a default argument and carries no information.

My working hypothesis is about timing. A norepinephrine record's END atom carries its rate into the next hour's first bundle. If that holds, MAP readings that directly follow a norepinephrine bundle should carry much more decodable rate information than the pooled R2 of 0.17 suggests. That would make a gated head channel for norepinephrine cheaper than I assumed.

To test it, I will split arterial MAP readings with norepinephrine at lag 1 by whether the preceding bundle held norepinephrine. For each split I will measure:
- cross-validated R2 and slope of decoding the rate and the dose surprise, from the full state and from the head input;
- the model's residual slope on the surprise, which is the factual deficit.

Clearly higher R2 in the "previous bundle had norepinephrine" split would confirm the hypothesis. Similar low values in both splits would refute it.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
m = (RD2.event.values=='map_arterial') & (RD2.ne_1.values>0)
pr = RD2.prev_row.values.astype(int)
pne = STALL.has_ne.values[pr].astype(bool)
res = (RD2.observed - RD2.pred0).values
print(m.sum(), (m&pne).sum(), round(RD2.ne_1[m].mean(),3), round(RD2.ne_1[m].std(),3), round(RD2.D_ne_1[m].std(),3), round(res[m].std(),2))
def r2k(X, y, g, a=10.0):
    Xs = (X - X.mean(0))/(X.std(0)+1e-6)
    p = cross_val_predict(Ridge(alpha=a), Xs, y, groups=g, cv=GroupKFold(5))
    return round(1-((y-p)**2).mean()/y.var(),3), round(np.cov(p,y)[0,1]/y.var(),3)
for nm, sub in [('all', m), ('pne', m&pne), ('nopne', m&~pne)]:
    g = RD2.stay_id.values[sub]; out_ = [nm, int(sub.sum())]
    for tgt in ['ne_1','D_ne_1']:
        y = RD2[tgt].values[sub]
        out_ += [tgt, r2k(XALL[pr[sub]], y, g, 30.0), r2k(WC0[sub], y, g, 10.0)]
    d = RD2.D_ne_1.values[sub]; out_.append(round(np.polyfit(d, res[sub], 1)[0],2))
    print(*out_)
```