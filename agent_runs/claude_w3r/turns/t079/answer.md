The band means of the sliding scale are clean: 0, 1.0, 2.0, 3.0, 4.04 and 6.17 u/h by glucose band. The dataset has all 1,865 glucose_finger readings, about two thirds of them preceded by an insulin infusion in the previous hour.

Next I fit grouped-CV ridge probes, grouped by stay, from the value head's hidden activations to:
- this hour's rate r_s;
- the band mean m;
- the deviation r_s − m;
- the previous hour's rate.

For comparison I fit the same probes from the full stored state.
- A deviation R² above about 0.6 from the hidden layer would make a head-only tilt viable.
- An R² near 0.2 would mean I need to build a cleaner dose channel first.

```python
def cvr2(X, y, groups, alphas=(1.0,10.0,100.0)):
    best=-9
    for a in alphas:
        pred=np.zeros_like(y)
        for trn,tst in GroupKFold(5).split(X,y,groups):
            mdl_=Ridge(alpha=a).fit(X[trn],y[trn]); pred[tst]=mdl_.predict(X[tst])
        r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum(); best=max(best,r2)
    return round(best,3)
DH['dev']=DH.rs-DH.m
ok_=~DH.gl.isna().values
Xst = np.array([S_['x'][head_inputs(r)[0]] for r in fsa.itertuples()])
for tgt in ['rs','m','dev','rs2']:
    y=DH[tgt].values[ok_]
    print(tgt, 'hidden R2', cvr2(FH[ok_], y, DH.stay_id.values[ok_]), 'state R2', cvr2(Xst[ok_], y, DH.stay_id.values[ok_]))
on_=ok_&(DH.m.values>0)
print('dev within on-band rows: hidden', cvr2(FH[on_], DH.dev.values[on_], DH.stay_id.values[on_]), 'state', cvr2(Xst[on_], DH.dev.values[on_], DH.stay_id.values[on_]), 'n', on_.sum(), 'sd dev', round(DH.dev.values[on_].std(),3))
```