All the kernel variables survived: states for 400 stays, the reading table with lagged dose and dose-surprise columns, and the offline head inputs. Next I measure how much of the true lag-1 dose-surprise effect the original model misses in its factual predictions. I regress the factual residual (observed minus predicted) of glucose readings on the lagged insulin surprise.

The multiplier noise is random, so a well-trained model should leave no slope. If the slope is close to the true minus-two-per-unit, the model ignores the surprise even factually, and a centred edit should also improve factual NLL. If the slope is near zero, the model already uses the surprise factually and only fails under extrapolation.

```python
ev = RD2['event'].astype(str)
print(ev.value_counts().to_dict())
r = RD2
print('res check', float(np.nanmax(np.abs((r['observed']-r['pred0'])-r['res']))))
from numpy.linalg import lstsq
def reg(mask, cols, y='res'):
    d = r.loc[mask, cols+[y]].dropna()
    X = np.column_stack([np.ones(len(d))]+[d[c].values for c in cols])
    b = lstsq(X, d[y].values, rcond=None)[0]
    e = d[y].values - X@b
    se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*e.var())
    return len(d), np.round(b[1:],2), np.round(se[1:],2)
for code in sorted(ev.unique()):
    if 'gluc' in code:
        m = (ev==code)
        print(code, reg(m, ['eps_ins_1','eps_ins_2','eps_ins_3']))
        print(code, 'w/ exp', reg(m, ['eps_ins_1','eps_ins_2','ins_exp_1','ins_exp_2']))
```