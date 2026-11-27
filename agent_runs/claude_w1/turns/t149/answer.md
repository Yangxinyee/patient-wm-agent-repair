The real model's teacher-forced output for 100 training stays is in. Reading shares are finger 32%, lab 11%, arterial 23% and cuff 34%; the per-reading NLL is 1.598.

Before requesting more passes I'll analyse this batch. For each glucose_finger and MAP_arterial row I'll:
- regenerate my replica's head inputs (exact core, factual doses);
- merge with the tool's predicted means;
- regress the true residuals on lagged rule innovations (`NUm`/`NUNm`, aligned for training stays);
- regress them on the decoded dose minus the rule expectation (`Dm − NUm`);
- check how predictable they are from x.

How to read the outcome:
- Insulin innovation coefficients near −1.5 and (decoded − expected) coefficients of the same sign: a decoded-dose correction would help factual fit.
- Wrong-sign or zero coefficients again (as on setA for MAP): the decoders must be redesigned to be orthogonal to the residual-predictive directions.

```python
F4 = pd.read_csv('/work/fwd_4/factual_original.csv')
st4 = [1010020000 + i for i in range(100)]
print('overlap with SAMP used', np.mean(np.isin(st4, list(used))))
S4 = []
for sv in st4: S4.extend(gen_samples(tr[tr.stay_id == sv], None, False))
S4d = pd.DataFrame([dict(code=t[0], stay=t[1], hour=t[2], cur=t[4], l1=t[5]) for t in S4]); X4 = np.stack([t[8] for t in S4]).astype(np.float64)
m4 = S4d.merge(F4.rename(columns={'stay_id':'stay','event':'code'}), on=['stay','hour','code'], how='left')
print('unmatched', m4.observed.isna().sum(), 'obs==cur', np.mean(np.isclose(m4.observed, m4.cur)))
m4['res'] = m4.observed - m4.predicted_mean
ix4 = (m4.stay.values - 1010000000).astype(int); h4 = m4.hour.values.astype(float)
for cd_, NU_, D_ in [('glucose_finger', NUm, Dm), ('map_arterial', NUNm, Rm)]:
    sel = (m4.code.values == cd_) & (h4 >= 1) & m4.res.notna().values
    Ld_ = lagmat(D_, ix4[sel], h4[sel], 3); Ln_ = lagmat(NU_, ix4[sel], h4[sel], 3); Pi_ = Ld_ - Ln_
    y0 = m4.res.values[sel]; dh_ = DEC[cd_].predict(X4[sel])
    print(cd_, 'n', sel.sum(), 'lag1 match', np.mean(np.isclose(Ld_[:,0], m4.l1.values[sel], atol=1e-3)).round(3), 'res mean %.2f sd %.2f' % (y0.mean(), y0.std()))
    for lab_, Z_ in [('nu', Ln_), ('dhat-pi', dh_ - Pi_), ('dhat,pi', np.c_[dh_, Pi_])]:
        b_, se_ = ols_cl(y0, Z_, ix4[sel]); print('   res ~', lab_, b_[1:].round(2).tolist(), 'se', se_[1:].round(2).tolist())
    # predictability of tool residual from x: CV ridge on top PCs
    Xs_ = X4[sel] - X4[sel].mean(0); U_, S__, Vt_ = np.linalg.svd(Xs_, full_matrices=False); Z20 = Xs_ @ Vt_[:30].T
    us_ = np.unique(ix4[sel]); fo_ = {sv: i % 5 for i, sv in enumerate(us_)}; fk_ = np.array([fo_[sv] for sv in ix4[sel]]); p_ = np.zeros(len(y0))
    for f_ in range(5): p_[fk_ == f_] = Ridge(alpha=10.0).fit(Z20[fk_ != f_], y0[fk_ != f_]).predict(Z20[fk_ == f_])
    print('   CV R2 res ~ 30 PCs of x', round(1 - ((y0 - p_)**2).mean() / y0.var(), 4))
```