The norepinephrine one-step coefficients settle the NE gap:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| true (all 312k train arterial readings, SE 0.08) | +2.89 | +2.14 | +0.84 | −0.12 | −0.66 | −0.87 |
| truth on the 500 states stays | +3.26 | +2.55 | +0.66 | −0.25 | −0.84 | −0.21 |
| model on the 500 states stays (SE 0.09) | +0.26 | −0.29 | −0.73 | −0.90 | −0.66 | −0.30 |

The truth on my selected stays matches the full train set, so selection bias is mild here. The model essentially ignores norepinephrine on MAP: Δβ at lag 1 is about +3.

Glucose readings behave like a persistent, nearly unit-root series. The one-step insulin coefficients follow the pattern expected when the drug's effect is transient on top of that: negative at lags 1–2, positive later. That means a lag-1-only correction accumulates in the rollout. For a sustained dose change ΔD, the shift after n hours is about a·n·ΔD. Matching the true average over hours 13–24 (about −9 per unit of sustained ΔD) needs a ≈ −1.4. That is close to the bound of about 1.45 at which the correction costs no factual NLL. MAP is more mean-reverting, so the NE fix will be partial.

Now I build the read-outs that the edit will use:
1. **Insulin:** one shared direction on `with_code` for glucose rows (finger and lab), with per-(code, last-bin) offsets, fitted to the lag-1 insulin deviation.
2. **NE:** the same structure for MAP rows (arterial and cuff), fitted to the lag-1 NE deviation.
3. For each, the cross-validated R² and gain, so I can rescale to unit gain.
4. A gate direction separating glucose codes from MAP codes, with its margins.

If the joint read-outs keep R² near 0.44 (insulin) and 0.18 (NE), and the gate leaves a clear margin, I build the edit in the next run.

```python
from sklearn.linear_model import Ridge
def fit_readout(mask, ycol, codes_):
    Rm = RA_[mask].reset_index(drop=True); Wm = WA_[mask]
    cl = np.array([codes_.index(int(c))*17 + int(l) for c, l in zip(Rm.code.values, Rm.lvb.values)])
    Dm_ = np.eye(17*len(codes_))[cl]
    F_ = np.c_[Wm, Dm_]; y = Rm[ycol].values; grpm = Rm.stay.values
    rcv = RidgeCV(alphas=np.logspace(-2,4,14)).fit(F_, y); alpha = rcv.alpha_
    pred = cross_val_predict(Ridge(alpha=alpha), F_, y, cv=GroupKFold(5), groups=grpm)
    gain = np.polyfit(y, pred, 1)[0]; r2 = 1-np.mean((y-pred)**2)/y.var()
    per = {ev: (1-np.mean((y[m_]-pred[m_])**2)/y[m_].var(), np.polyfit(y[m_], pred[m_], 1)[0]) for ev in Rm.event.unique() for m_ in [(Rm.event==ev).values]}
    m = Ridge(alpha=alpha).fit(F_, y)
    w = m.coef_[:192]; bdum = m.coef_[192:].reshape(len(codes_), 17); w0 = m.intercept_
    return dict(w=w, b=bdum, w0=w0, gain=gain, r2=r2, alpha=alpha, per=per, codes=codes_, pred_cv=pred, y=y, rows=Rm)
mins = RA_.event.isin(['glucose_finger','glucose_lab']).values
mne = RA_.event.isin(['map_arterial','map_cuff']).values
RO_ins = fit_readout(mins, 'dev1', [6, 10])
RO_ne = fit_readout(mne, 'nedev1', [7, 5])
for nm, ro in [('insulin', RO_ins), ('NE', RO_ne)]:
    print(nm, 'alpha', ro['alpha'], 'CV R2', round(ro['r2'],3), 'gain', round(ro['gain'],3), 'per event (R2, gain):', {k: (round(v[0],3), round(v[1],3)) for k, v in ro['per'].items()}, '|w|', round(np.linalg.norm(ro['w']),3))
# gate direction separating glucose codes from MAP codes on mean wc
gl_rows = WA_[mins]; mp_rows = WA_[mne]
gdir = gl_rows.mean(0) - mp_rows.mean(0)
Sw = np.cov(np.r_[gl_rows - gl_rows.mean(0), mp_rows - mp_rows.mean(0)].T) + 1e-3*np.eye(192)
glda = np.linalg.solve(Sw, gdir); glda /= np.linalg.norm(glda)
for nm, gv in [('meandiff', gdir/np.linalg.norm(gdir)), ('lda', glda)]:
    s_all = WA_ @ gv
    print(nm, {ev: np.round(np.percentile(s_all[(RA_.event==ev).values], [0, 1, 50, 99, 100]), 2).tolist() for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
    print('   readout ins on MAP rows range:', np.round(np.percentile(WA_[mne] @ RO_ins['w'], [0, 100]), 2), ' readout ne on glucose rows range:', np.round(np.percentile(WA_[mins] @ RO_ne['w'], [0, 100]), 2))
```