With the fitted layout, the held-out error of my head rebuild is 1.3 units: 1.5–1.7 on glucose, 0.8–1.1 on MAP. The fit-half error is similar, so this is not overfitting. Part of what remains is probably the model's random z draws. This is good enough for finding unused units, setting gate margins and rough offline cost checks.

Now the main quantity: how much dose effect does the original model miss, given everything it already conditions on? On these 5,521 factual readings I regress the model's prediction error (observed − predicted) on the random parts of the doses 1–4 hours earlier:
- **Insulin:** dose minus its expected value given the last glucose and whether insulin was on.
- **Norepinephrine:** rate minus its expected value given the previous rate and the MAP band.

I also regress the model's own predicted means on the same residuals, to get its own dose sensitivity.

Clearly negative error coefficients for insulin on glucose, and positive ones for norepinephrine on MAP, would confirm that the model underweights both drugs. Their size gives the kernel my correction must supply. The cross terms should come out near zero.

```python
def lastmap(r_, t_):
    a_ = ma[r_, t_]; return a_ if not np.isnan(a_) else mc[r_, t_]
# insulin residual (all 30000 stays, hours 0..24)
Hh = np.arange(25)
d_all = ins_all; on_prev_all = np.concatenate([np.zeros((30000, 1)), (ins_all[:, :-1] > 0)], 1)
gl_all = np.nan_to_num(glast[:, :25], nan=150.0)
bnd = np.digitize(gl_all, [130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 300])
key = bnd * 2 + on_prev_all.astype(int)
mean_tab = np.zeros(26); 
for k_ in range(26): mean_tab[k_] = d_all[key == k_].mean() if (key == k_).any() else 0
eps_ins = d_all - mean_tab[key]
# norepi residual: d_t vs d_{t-1} x MAP band
mlast_all = np.where(~np.isnan(ma[:, :25]), ma[:, :25], mc[:, :25]); mlast_all = np.nan_to_num(mlast_all, nan=70.0)
mb = np.digitize(mlast_all, [56, 60, 64, 66, 70, 74, 76, 80, 85, 90])
ne_prev = np.concatenate([np.zeros((30000, 1)), ne_all[:, :-1]], 1)
eps_ne = np.zeros_like(ne_all)
for b__ in range(11):
    for pz in [0, 1]:
        m_ = (mb == b__) & ((ne_prev > 0) == pz)
        if m_.sum() < 10: continue
        Xr = np.column_stack([np.ones(m_.sum()), ne_prev[m_], ne_prev[m_] ** 2])
        co_, *_ = np.linalg.lstsq(Xr, ne_all[m_], rcond=None); eps_ne[m_] = ne_all[m_] - Xr @ co_
print('eps sd: ins %.3f ne %.3f' % (eps_ins.std(), eps_ne[ne_all > 0].std()))
rowi4 = M4.stay_id.map(sidx).values; th4 = M4.hour.values.astype(int)
def lagmat(E_, L=4):
    return np.column_stack([np.where(th4 - l_ >= 0, E_[rowi4, np.clip(th4 - l_, 0, 24)], 0.0) for l_ in range(1, L + 1)])
EI, EN = lagmat(eps_ins), lagmat(eps_ne)
err4 = M4.observed.values - M4.predicted_mean.values
for grp_, evs_ in [('glucose', ['glucose_finger', 'glucose_lab']), ('MAP', ['map_arterial', 'map_cuff'])]:
    m_ = M4.event.isin(evs_).values & (th4 >= 1)
    Xr = np.column_stack([np.ones(m_.sum()), EI[m_], EN[m_]])
    for nm_, y_ in [('model error', err4[m_]), ('model pred', M4.predicted_mean.values[m_]), ('observed', M4.observed.values[m_])]:
        co_, res__, *_ = np.linalg.lstsq(Xr, y_, rcond=None)
        s2 = np.sum((y_ - Xr @ co_) ** 2) / (len(y_) - Xr.shape[1]); se_ = np.sqrt(np.diag(s2 * np.linalg.inv(Xr.T @ Xr)))
        print(grp_, '%-11s' % nm_, 'ins lags1-4', np.round(co_[1:5], 2), '+-', np.round(se_[1:5], 2), '| ne lags1-4', np.round(co_[5:9], 2), '+-', np.round(se_[5:9], 2))
```