## Step 16: lab glucose is a selection-free outcome

**What the last result showed.**
- Lab glucose is drawn at random with probability 0.20 every hour. It does not depend on glucose, insulin or the hour, and the gaps between labs are geometric.
- Finger checks are hourly whenever the previous latest glucose was above 150; otherwise they happen at random 30% of the time.

**What this means.** Lab glucose is an unbiased outcome for multi-hour effects (about 150k readings).

**What I will do next.** Estimate the insulin impulse response h_m, the effect of one unit/h given m hours before a lab reading, holding the other doses fixed.
- Regress the lab reading at hour s on the doses at s−1 … s−K, instrumented by their surprises e (a just-identified two-stage least squares).
- Control only for the history at s−K, which comes before any of the instruments.
- Then the cumulative sum of h tells me how a sustained dose change builds up over hours.

**What would count as evidence.**
- h_1 ≈ −2 with later h_m decaying slowly would mean large cumulative effects: glucose has strong persistence.
- h_m near zero for m ≥ 2 would mean only a transient effect.

```python
def tsls(y, endog, instr, exog):
    # y (n,), endog (n,k), instr (n,k), exog (n,p) incl const
    Z = np.column_stack([instr, exog]); X = np.column_stack([endog, exog])
    # first stage projection
    PZX = Z @ la.lstsq(Z, X, rcond=None)[0]
    b = la.lstsq(PZX, y, rcond=None)[0]
    r = y - X@b
    # robust (HC0) se
    A = la.inv(PZX.T@PZX)
    meat = (PZX*(r[:,None]**2)).T @ PZX
    cov = A@meat@A
    return b, np.sqrt(np.diag(cov))
xs = x.sort_values(['stay_id','h']).copy()
grp = xs.groupby('stay_id')
K = 8
for m in range(0, K+2):
    xs[f'I_l{m}'] = grp.ins_rate.shift(m)
    xs[f'e_l{m}'] = grp.e_ins.shift(m)
    xs[f'lg_l{m}'] = grp.lg.shift(m)
def irf(xs, ycol, K, extra_ctrl=True):
    cols = [f'I_l{m}' for m in range(1,K+1)] + [f'e_l{m}' for m in range(1,K+1)] + [f'lg_l{K}', f'I_l{K+1}']
    d = xs[xs[ycol].notna()].dropna(subset=cols)
    endog = d[[f'I_l{m}' for m in range(1,K+1)]].values
    instr = d[[f'e_l{m}' for m in range(1,K+1)]].values
    lgK = d[f'lg_l{K}'].values
    ex = [np.ones(len(d)), lgK, np.maximum(lgK-150,0), np.maximum(lgK-180,0), np.maximum(lgK-220,0), d[f'I_l{K+1}'].values]
    exog = np.column_stack(ex)
    b, se = tsls(d[ycol].values, endog, instr, exog)
    return b[:K], se[:K], len(d)
for K_ in [4, 8]:
    b, se, n = irf(xs, 'glucose_lab', K_)
    print("K=",K_, "n=",n)
    print("  h_m :", np.round(b,2).tolist())
    print("  se  :", np.round(se,2).tolist())
    print("  cumsum:", np.round(np.cumsum(b),2).tolist())
```