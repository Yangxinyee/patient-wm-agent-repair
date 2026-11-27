Run 135 printed the tuning panel layout. `Ptu3` is indexed by stay and hour −1..24 and has, per hour, the insulin rate, the norepinephrine rate `R`, the rule-expected rate `Rhat` and its innovation `nuN`. I generated 3,128 factual setA samples (glucose_finger and MAP_arterial rows) with my exact core replica.

Next I'll merge these with the tool's `F2` teacher-forced means. That gives the real model's residuals. I'll regress them on:
- rule innovations;
- decoded doses minus rule expectations.

How to read the outcome:
- Tool residuals loading negatively on insulin innovations and on (decoded − expected) with similar size: the wrong-sign result was a replica artefact, and the decoded-dose correction should be safe for the factual fit.
- A positive loading on the decoded dose here too: the head really reads its input differently, and I'll need another design (e.g., subtracting a decoded rule expectation).

For the insulin innovations I'll use `NUA`, after checking its alignment against the setA insulin doses.

```python
SAd = pd.DataFrame([dict(code=t[0], stay=t[1], hour=t[2], prev=t[3], cur=t[4], l1=t[5], l2=t[6], l3=t[7]) for t in SA])
XA = np.stack([t[8] for t in SA]).astype(np.float64)
mA_ = SAd.merge(F2.rename(columns={'stay_id':'stay','event':'code'}), on=['stay','hour','code'], how='left')
print('unmatched', mA_.observed.isna().sum(), 'obs==cur', np.mean(np.isclose(mA_.observed, mA_.cur)))
mA_['res'] = mA_.observed - mA_.predicted_mean
pt_ = Ptu3.reset_index()
def pmat(col):
    M_ = np.full((len(setA), 26), np.nan); sub_ = pt_[pt_.stay_id.isin(setA)]
    for sv, hh_, v_ in zip(sub_.stay_id, sub_.h, sub_[col]): M_[aix_[sv], int(hh_) + 1] = v_
    return np.nan_to_num(M_)
aix_ = {sv: i for i, sv in enumerate(setA)}
INSA, RA_, RhA, NUNA = pmat('insulin'), pmat('R'), pmat('Rhat'), pmat('nuN')
print('DA vs insulin panel', np.allclose(DA, INSA, atol=1e-3), 'corr NUA,INSA', np.corrcoef(NUA.ravel(), INSA.ravel())[0,1].round(3), 'NUNA vs R-Rhat', np.allclose(NUNA, RA_ - RhA, atol=1e-3))
ixA_ = np.array([aix_[sv] for sv in mA_.stay]); hA_ = mA_.hour.values.astype(float)
for cd_, NU_, D_ in [('glucose_finger', NUA, INSA), ('map_arterial', NUNA, RA_)]:
    sel = (mA_.code.values == cd_) & (hA_ >= 1) & mA_.res.notna().values
    Ld_ = lagmat(D_, ixA_[sel], hA_[sel], 3); Ln_ = lagmat(NU_, ixA_[sel], hA_[sel], 3); Pi_ = Ld_ - Ln_
    print(cd_, 'n', sel.sum(), 'lag1 match', np.mean(np.isclose(Ld_[:,0], mA_.l1.values[sel], atol=1e-3)).round(3), 'res mean %.2f sd %.2f' % (mA_.res[sel].mean(), mA_.res[sel].std()))
    dh_ = DEC[cd_].predict(XA[sel]); y0 = mA_.res.values[sel]
    for lab_, Z_ in [('nu', Ln_), ('dhat-pi', dh_ - Pi_), ('dhat', dh_), ('dhat,pi', np.c_[dh_, Pi_])]:
        b_, se_ = ols_cl(y0, Z_, ixA_[sel]); print('   res ~', lab_, b_[1:].round(2).tolist(), 'se', se_[1:].round(2).tolist())
```