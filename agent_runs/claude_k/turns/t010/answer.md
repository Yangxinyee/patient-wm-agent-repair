The last run failed on a syntax error from a stray line I added; statsmodels isn't available anyway. I'll rerun the same analysis without it and do the regression with numpy least squares. The plan is unchanged: compare paired finger/lab and arterial/cuff readings, then regress the hour-to-hour change in glucose_finger on the insulin rate with policy controls. A clear negative insulin coefficient that is stable to the controls would support a real drug effect.

```python
both = np.isfinite(gf) & np.isfinite(gl)
d = gf[both] - gl[both]
print("paired finger-lab: n", both.sum(), "mean diff", d.mean().round(2), "sd", d.std().round(2), "corr", np.corrcoef(gf[both], gl[both])[0,1].round(3))
bm = np.isfinite(ma) & np.isfinite(mc)
dm = ma[bm]-mc[bm]
print("paired art-cuff: n", bm.sum(), "mean diff", dm.mean().round(2), "sd", dm.std().round(2), "corr", np.corrcoef(ma[bm], mc[bm])[0,1].round(3))
rows=[]
for t in range(0,24):
    j=col(t)
    rows.append(pd.DataFrame({'t':t,'g0':gf[:,j],'g1':gf[:,j+1],'glast':glast[:,j],'ins':ins[:,j],'ins_prev':ins[:,j-1],
                              'ne':ne[:,j],'m':mlast[:,j],'stay':np.arange(len(ids))}))
R = pd.concat(rows, ignore_index=True)
Rv = R.dropna(subset=['g0','g1']).copy()
Rv['dg'] = Rv.g1-Rv.g0
def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))]+X); b,res,rk,sv = np.linalg.lstsq(X,y,rcond=None)
    r = y - X@b; s2 = r@r/(len(y)-X.shape[1]); cov = s2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
    return pd.DataFrame({'b':b,'se':se}, index=['const']+names)
print(ols(Rv.dg.values, [Rv.ins.values], ['ins']))
print(ols(Rv.dg.values, [Rv.ins.values, Rv.g0.values], ['ins','g0']))
# with flexible glucose controls (bins of g0)
gb = pd.get_dummies(pd.cut(Rv.g0,[0,100,120,140,150,160,170,180,190,200,210,220,240,260,300,1000]), drop_first=True).values.astype(float)
print(ols(Rv.dg.values, [Rv.ins.values, Rv.g0.values, Rv.ins_prev.values, Rv.ne.values]+[gb[:,i] for i in range(gb.shape[1])], ['ins','g0','ins_prev','ne']+[f'gb{i}' for i in range(gb.shape[1])]).iloc[:5])
```