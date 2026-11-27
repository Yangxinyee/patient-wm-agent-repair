z26 does not code the dose. At insulin bundles it is binary: about −1.52 for rates 1.0–1.5 U/h and +1.53 otherwise, at every glucose tier. It looks like a 1-bit feature of the common rate values, not a linear dose code, so the z route is dead for insulin.

The dose reaches the state through the action token, and the model's dose response is nonlinear. Before choosing a state-level lever, I need to know whether the nonlinearity arises before the state or after it. I project the state change across each insulin bundle onto the fitted innovation direction and tabulate it by recorded rate within glucose tier.
- If that projection grows roughly linearly with rate, the encoder output is fine. The flattening is then downstream, and amplifying the dose-specific state component could work.
- If the projection itself flattens at high rates, the saturation is in the encoder and no state-level scaling fixes it.

```python
def proj_table(Sx, label):
    x = Sx['x']; sidx = Sx['stay_id']
    same = np.r_[False, sidx[1:]==sidx[:-1]]
    sel = same[ins_b] & ok
    rr = ins_b[sel]
    D_ = x[rr] - x[rr-1]
    A_ = np.column_stack([np.ones(sel.sum()), innov[sel], exp_rate[sel], (gsrc_b[sel]-200)/50, hn3[rr].astype(float)])
    c_, *_ = np.linalg.lstsq(A_, D_, rcond=None)
    res = {}
    for nm_, sl in [('out', slice(0,192)), ('h', slice(192,960))]:
        r_ = c_[1, sl]/np.linalg.norm(c_[1, sl])
        p_ = D_[:, sl] @ r_
        df_ = pd.DataFrame({'p': p_, 'rate': rate[sel], 'tier': tk[sel], 'ne': hn3[rr]})
        tab = df_.groupby(['tier', pd.cut(df_.rate, [0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 4.25, 12])], observed=True)['p'].mean().unstack(0).round(2)
        print(label, nm_); print(tab.to_string())
        res[nm_] = r_
    return res
r_orig = proj_table(So, 'orig')
```