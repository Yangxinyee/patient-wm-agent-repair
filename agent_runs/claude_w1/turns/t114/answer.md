Insulin is never given in an hour without a glucose reading, so the policy reference is needed only at reading hours. I'll take it as E[dose | rounded glucose, previous on/off] tabulated from the 30,000 training stays, and set innovation = dose − that.

Now the feasibility test. On 3,000 training stays I'll generate factual head inputs at finger-glucose readings, plus paired copies with each insulin infusion rescaled by an independent log-normal factor (sd 0.5). Regressing the paired head-input differences on the dose changes at lags 1–10 gives the dose directions. I'll then build the minimum-variance decoder with response fixed to my deconvolved insulin impulse response. I'll judge it by its factual error relative to the true innovation term T = Σ β_k ν_{t−k}. An error sd well below sd(T) means a head-only edit is viable and would even improve factual fit. An error sd at or above sd(T) means I need an upstream change.

```python
idx = p2.index; hv = idx.get_level_values(1).values; sv = idx.get_level_values(0).values
assert (hv[:26] == np.arange(-1,25)).all() and (np.diff(sv[::26]) > 0).all()
S_ord = sv[::26]; six = {s:i for i,s in enumerate(S_ord)}
Dm = p2.D.fillna(0).values.reshape(-1, 26)
Gi = np.round(p2.Gany.values); po = p2.prev_on.values
tab = pd.DataFrame({'G': Gi, 'po': po, 'D': p2.D.fillna(0).values}).dropna().groupby(['G','po']).D.mean()
Epi = np.where(np.isnan(Gi), 0.0, tab.reindex(pd.MultiIndex.from_arrays([Gi, po])).values)
Epi = np.nan_to_num(Epi); NUm = (p2.D.fillna(0).values - Epi).reshape(-1, 26)
print('nu sd (reading hours, D>0 or G>140): %.3f' % NUm.ravel()[(~np.isnan(Gi)) & (Gi > 140)].std())
hins = np.array([-2.08,-2.41,-2.16,-1.79,-1.26,-0.95,-0.77,-0.63,-0.55,-0.43]); K = 10
rng = np.random.default_rng(11); st3 = rng.choice(S_ord, 3000, replace=False)
trs = tr[tr.stay_id.isin(st3)].copy()
mi = trs.event.eq('insulin').values; fac = np.exp(0.5*rng.standard_normal(mi.sum()))
trp = trs.copy(); trp.loc[mi, 'dose'] = trp.loc[mi, 'dose'].values*fac; trp.loc[mi, 'rate'] = trp.loc[mi, 'rate'].values*fac
dDm = np.zeros_like(Dm); ii = np.array([six[s] for s in trs.stay_id.values[mi]]); hh = np.floor(trs.hour.values[mi]).astype(int)+1
np.add.at(dDm, (ii, hh), trs.dose.values[mi]*(fac-1))
fsF = gen_samples(trs, pert=False); fsP = gen_samples(trp, pert=False)
keyF = [(r[0], r[1], r[2]) for r in fsF]; assert keyF == [(r[0], r[1], r[2]) for r in fsP]
m = np.array([r[0]=='glucose_finger' for r in fsF]) & np.array([r[2] >= 0 for r in fsF])
xF = np.stack([r[8] for r in fsF])[m]; xP = np.stack([r[8] for r in fsP])[m]
rs = np.array([six[r[1]] for r in fsF])[m]; rt = np.array([int(r[2]) for r in fsF])[m]
def lagm(M_, rs, rt, K):
    out = np.zeros((len(rs), K))
    for k in range(1, K+1):
        c = rt - k + 1; ok = c >= 0; out[ok, k-1] = M_[rs[ok], c[ok]]
    return out
LdD = lagm(dDm, rs, rt, K); Lnu = lagm(NUm, rs, rt, K)
dX = xP - xF; A = np.linalg.lstsq(LdD, dX, rcond=None)[0]           # K x 192
r2a = 1 - ((dX - LdD@A)**2).sum()/((dX - dX.mean(0))**2).sum()
Sig = np.cov(xF.T); Sig += 1e-4*np.trace(Sig)/192*np.eye(192); Si = np.linalg.inv(Sig)
Gm = A@Si@A.T; w = Si@A.T@np.linalg.solve(Gm, hins)
T = Lnu@hins; yhat = xF@w
e = (yhat - yhat.mean()) - (T - T.mean())
print('rows %d | R2 of dX on lagged dD: %.3f | |a_k| first4:' % (len(xF), r2a), np.linalg.norm(A, axis=1)[:4].round(3))
print('MVDR noise sd %.2f | factual: sd(T) %.2f, sd(w.x) %.2f, err sd %.2f, corr %.3f' % (np.sqrt(hins@np.linalg.solve(Gm, hins)), T.std(), yhat.std(), e.std(), np.corrcoef(yhat, T)[0,1]))
print('pert calib: slope of w.dX on dD.h = %.3f' % np.polyfit(LdD@hins, dX@w, 1)[0])
```