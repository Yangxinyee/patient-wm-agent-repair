Regression adjustment finds no one-hour or two-hour effect of insulin. The coefficient is +0.037 ± 0.041 mg/dL per u/h at one hour and +0.035 ± 0.045 at two, whether or not tier dummies are included. The three-hour fit broke numerically because the spline basis became singular on that subset, so I am ignoring it. With random dose noise of about 33% inside each tier, these estimates are tight. Any immediate insulin effect is within about ±0.1 mg/dL per u/h, much smaller than my rough reading of the threshold jumps, which mistook mean reversion for a jump.

Insulin could still act with a delay or slowly build up, so next I will look further out. I will use a numerically safer estimator: fixed effects for the exact reading value (demeaning within each integer reading), with standard errors clustered by stay. I will estimate the effect of dose at t on readings 1–8 hours later, plus a version that also controls for the previous hour's dose. Coefficients near zero at every horizon would point to insulin truly having no effect on glucose in this ICU. A pattern growing with horizon would point to a delayed effect.

```python
def fe_slope(y, x, g, cl):
    df = pd.DataFrame({'y':y,'x':x,'g':g,'cl':cl}).dropna()
    df['yd'] = df.y - df.groupby('g').y.transform('mean')
    df['xd'] = df.x - df.groupby('g').x.transform('mean')
    sxx = (df.xd**2).sum(); b = (df.xd*df.yd).sum()/sxx
    e = df.yd - b*df.xd
    s = (df.xd*e).groupby(df.cl).sum()
    se = np.sqrt((s**2).sum())/sxx
    return b, se, len(df)
D = Q[Q.hr>=0].copy()
for k in range(1,9):
    D[f'Rf{k}'] = D.groupby('stay').glucose_finger.shift(-k) if False else np.nan
# compute future readings from P directly (P has all hours)
gfP = P.glucose_finger.groupby(level=0)
fut = {k: gfP.shift(-k) for k in range(1,10)}
for k in range(1,10):
    D[f'Rf{k}'] = fut[k].reindex(D.index).values
g = D.glucose_finger.round().astype(int).values
for k in range(1,10):
    b,se,n = fe_slope((D[f'Rf{k}']-D.glucose_finger).values, D.ins.values, g, D.stay.values)
    print(f"k={k}: effect of I_t on R_t+{k}: {b:.3f} +- {se:.3f}  n={n}")
```