The most efficient direction for δ moves the next finger-glucose prediction by about 0.099 mg/dL per unit norm, roughly 65 times more than the nuG direction. A 2 mg/dL shift then needs a δ of norm about 20 against slot norms of about 860, which is feasible.

The model's carry-forward of such a shift should fall between the reading-only 0.55 and the full-belief 0.96. The ideal is about 0.72, which with a per-dose gain near 2.6 would match the true insulin response curve, so I will measure and tune this with forward passes.

Next I'll fit the same kind of readout for all four predicted means (finger, lab, arterial, cuff). Then I'll solve for a minimum-norm δ_I that moves finger and lab equally and leaves both MAP predictions unchanged, and a δ_N doing the reverse, using the Jacobian.

```python
def pred_readout(ev):
    f_ = fa[fa.event==ev][['stay_id','hour','predicted_mean']].reset_index(drop=True)
    q_ = f_.copy(); q_['hq'] = q_.hour - 1e-6
    m2 = pd.merge_asof(q_.sort_values('hq'), ds_[['stay_id','hour','row']].rename(columns={'hour':'hb'}).sort_values('hb'), left_on='hq', right_on='hb', by='stay_id', direction='backward').dropna(subset=['row'])
    r_ = m2.row.astype(int).values
    X_ = np.c_[hv[r_], np.log(np.maximum((m2.hour - m2.hb).values*3600,1.0))]
    y_ = m2.predicted_mean.values
    r2cv = cvr2(X_, y_, m2.stay_id.values, alpha=1.0)
    return Ridge(alpha=1.0).fit(X_, y_).coef_[:192], r2cv, len(y_)
WR = {}
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    w_, r_, n_ = pred_readout(ev); WR[ev] = w_; print(ev, 'n', n_, 'CV R2 %.3f' % r_, '|w| %.2f' % np.linalg.norm(w_))
Wst = np.stack([WR[e] for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']])  # 4x192
Gm = np.einsum('ki,ij->kj', Wst, Jm_.mean(0))   # 4x192: pred change per unit delta (mean Jacobian)
def mindelta(target):
    return Gm.T @ np.linalg.solve(Gm @ Gm.T, target)
dI = mindelta(np.array([1.,1.,0.,0.])); dN = mindelta(np.array([0.,0.,1.,1.]))
print('|delta_I| per mg/dL %.2f  |delta_N| per mmHg %.2f' % (np.linalg.norm(dI), np.linalg.norm(dN)))
pb = np.einsum('ki,nij,j->nk', Wst, Jm_, dI)
print('delta_I per-bundle effects mean', np.round(pb.mean(0),3), 'sd', np.round(pb.std(0),3))
```