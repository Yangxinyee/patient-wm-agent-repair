The first effect estimates are in:
- A 1 u/h higher insulin rate at hour t, assigned at random, lowers finger glucose by about 2.0 mg/dL at t + 1, 2.25 at t + 2 and 1.9 at t + 3.
- The effect then fades: −1.3, −0.9, −0.6, −0.3 and −0.1 over lags 4–8.
- The standard errors are about 0.05, so insulin really does lower glucose in this ICU, but modestly.

These are total effects. They include the policy's feedback: lower glucose leads to lower later doses, which counteracts the effect. The dose_edit readout holds later doses fixed (recorded × 2), so I need the controlled direct effect of each dose.

Next I'll estimate it with a linear structural nested mean model (g-estimation). For lag ℓ the moment condition is

E[r_s · f_{s+ℓ}] = Σ_{j=1..ℓ} γ_j · E[r_s · rate_{s+ℓ−j}],

where r_s is the randomised rate residual. Solving it recursively gives the per-dose effect γ_j at each lag with later doses held fixed. If γ_j is about −2 and decays more slowly than the total effect, the readout for x2 is roughly −Σγ × the average dose.

```python
def to_grid(P_):
    H = np.arange(-1,25)
    sid = P_.stay_id.unique()
    Pi = P_.set_index(['stay_id','h'])
    def grid(col):
        return Pi[col].unstack().reindex(index=sid, columns=H).values
    return sid, H, grid
sid, H, grid = to_grid(P)
F = grid('glucose_finger'); L = grid('glucose_lab'); R = np.nan_to_num(grid('insulin'))
Gn = np.where(np.isnan(F), L, F)
# residual: rate - mean rate given integer g (among insulin-given hours)
ins_given = R>0
gi = np.where(ins_given, np.round(Gn), -1).astype(int)
mean_by_g = pd.Series(R[ins_given]).groupby(gi[ins_given]).mean()
Mg = np.zeros_like(R); Mg[ins_given] = mean_by_g.reindex(gi[ins_given]).values
Rres = np.where(ins_given, R - Mg, 0.0)
print("resid var", Rres[ins_given].var())
nS, nH = R.shape
maxlag = 12
gam = np.zeros(maxlag+1)
for l in range(1, maxlag+1):
    # sum over s, with s+l < nH and F[:, s+l] observed
    lhs = 0.0; coefs = np.zeros(l+1)
    for s_ in range(0, nH-l):
        y = F[:, s_+l]; m = ~np.isnan(y) & ins_given[:, s_]
        r = Rres[m, s_]
        yc = y[m] - 0  # no centering needed in expectation, but reduce variance below
        lhs += (r*(yc - np.nanmean(y))).sum()
        for j in range(1, l+1):
            coefs[j] += (r*R[m, s_+l-j]).sum()
    gam[l] = (lhs - (coefs[1:l]*gam[1:l]).sum())/coefs[l]
    print(l, round(gam[l],3), " cumulative:", round(gam[1:l+1].sum(),3))
```