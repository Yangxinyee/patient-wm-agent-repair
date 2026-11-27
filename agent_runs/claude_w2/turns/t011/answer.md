The insulin policy is a nearly deterministic sliding scale on the fresh glucose_finger at hour t:
- no insulin below 150 (150 itself is a 50/50 rounding tie);
- a base rate of about 1.04 (sd 0.35) for 150-179;
- about 2.09 (sd 0.65) from 180, with another tie at 180, and further steps higher up.

Glucose_finger is also always measured when the last value was above 150. The rare insulin starts without a fresh finger reading (0.9-8.7%) must be triggered by something else, possibly a lab glucose.

The within-band spread of rates (0.5/1.0/1.5 at the same glucose) is the key question. If it is independent of the patient's state, it is effectively a randomised experiment and gives the true effect. Next I'll test whether that within-band rate variation correlates with anything observed: the previous rate, the previous glucose, the glucose trend, lab glucose, MAP, NE, or the hour. I'll also check what triggers insulin when there is no fresh finger reading.

```python
y = x[x.gf.notna() & x.ins.notna()].copy()
y['band'] = pd.cut(y.gf, [150,180,220,250,300,2000], right=False)
y['gf_prev'] = y.groupby('stay_id').gf_ff.shift(1)  # careful: computed within subset; recompute on full
x['gf_prevff'] = x.groupby('stay_id').gf_ff.shift(1)
y = y.join(x[['gf_prevff']])
import statsmodels.api as sm if False else None
from sklearn.linear_model import LinearRegression
for b, d in y.groupby('band', observed=True):
    d = d[(d.gf>b.left+1)&(d.gf<b.right-1)].dropna(subset=['gf_prevff'])
    X = pd.DataFrame({'gf':d.gf, 'ins_prev':d.ins_prev, 'gf_prev':d.gf_prevff, 'h':d.h, 'nor0':d.nor0,
                      'ma':d.ma_ff.fillna(d.mc_ff), 'gl_fresh':d.gl.notna().astype(float)})
    X = X.fillna(X.mean())
    lr = LinearRegression().fit(X, d.ins)
    print(b, len(d), 'R2', round(lr.score(X,d.ins),4), dict(zip(X.columns, np.round(lr.coef_,4))))
    print('   rate dist', d.ins.value_counts(normalize=True).sort_index().round(3).to_dict())
# triggers when no fresh gf
z = x[x.gf.isna() & x.ins.notna()]
print(len(z), z.gl.notna().mean(), z[['gl','gl_ff','gf_ff']].describe().to_string())
```