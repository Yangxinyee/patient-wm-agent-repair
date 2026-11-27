The panel builder reproduces `Ptr2` exactly. Now the replica-bias check on setA's 100 tuning stays. I'll compute insulin innovations with the training-set policy table. Then I'll regress (tool mean − my mean) on the innovations at lags 1–3 for the finger readings. Coefficients near zero (within ~±0.3 mg/dL per u/h) mean my local means can stand in for the tool's when estimating the missing dose effect.

```python
def nu_of(D, Gany, pon):
    Gi_ = np.round(Gany); key = pd.MultiIndex.from_arrays([Gi_.ravel(), pon.ravel()])
    E = np.nan_to_num(tab.reindex(key).values).reshape(D.shape); E[~np.isfinite(Gany)] = 0.0
    return D - E, E
stA, ixA, DA, GA, pA = build_panel(tu[tu.stay_id.isin(setA_ids)]); NUA, EA = nu_of(DA, GA, pA)
mu, _ = head_mean(rowsA); dfA['mine'] = mu
mg_ = dfA.merge(f2, on=['stay_id','hour','event'], how='inner')
gf = mg_[mg_.event.eq('glucose_finger') & (mg_.hour >= 0)]
rsA = gf.stay_id.map(ixA).values; rtA = gf.hour.values.astype(int)
LnuA = lagm(NUA, rsA, rtA, 3); dtool = (gf.predicted_mean - gf.mine).values
Xr = np.column_stack([np.ones(len(gf)), LnuA]); bet, res_, *_ = np.linalg.lstsq(Xr, dtool, rcond=None)
se = np.sqrt(np.diag(np.linalg.inv(Xr.T@Xr)) * (dtool - Xr@bet).var())
print('n %d | tool-mine mean %.2f sd %.2f | coef on nu lag1..3:' % (len(gf), dtool.mean(), dtool.std()), bet[1:].round(3), 'se', se[1:].round(3))
print('corr(tool-mine, tool mean) %.3f' % np.corrcoef(dtool, gf.predicted_mean)[0,1])
```