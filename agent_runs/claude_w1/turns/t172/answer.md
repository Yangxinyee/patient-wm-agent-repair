The setA readings split into finger 34%, lab 10%, arterial 26% and cuff 29%, with residual sds of 14.9, 12.0, 5.2 and 7.9. A 0.005-nat budget split between finger and arterial therefore allows roughly 1.8 mg/dL rms of uncorrelated extra error on finger and 0.7 mmHg on arterial.

Now I test whether the head-level route fits that budget on setA, using the stored perturbed-dose decoders and exact tool residuals. For a lag-1 correction δ I compute E[δ²] − 2E[rδ] in three versions:
1. an oracle using the true innovation;
2. the decoded dose minus the true policy expectation, which isolates decoder noise;
3. the decoded dose minus a policy expectation that depends only on the last-reading bin. This is the implementable version and keeps the response to the dose level.

Small or negative predicted NLL changes for version 3 would make the head-level edit viable. Large costs would mean the decoder noise has to be reduced first.

```python
GU = dict(tuple(tu[tu.stay_id.isin(setA)].groupby('stay_id')))
rowsA = [r for s in setA for r in gen_samples(GU[s], pert=False)]
codesA = np.array([str(r[0]) for r in rowsA]); iiA = idxU[[r[1] for r in rowsA]].values; ccA = np.round([r[2] for r in rowsA]).astype(int)+1
XA_ = np.stack([r[8] for r in rowsA])
F2i = F2df.set_index(['stay_id','hour','event'])['res']
rA = F2i.reindex(pd.MultiIndex.from_arrays([[r[1] for r in rowsA], [float(r[2]) for r in rowsA], list(codesA)])).values
print('rows', len(rowsA), 'matched', np.isfinite(rA).sum())
def lagU(M, k): return np.where(ccA-k>=0, M[iiA, np.maximum(ccA-k,0)], 0.)
lastbinA = np.array([lastbin_of(r[0], r[3]) for r in rowsA])
for cd, drug, Dm_, pim, c, sig, frac in [('glucose_finger','ins',OU['ins'],pIU,-2.5,14.86,0.34), ('map_arterial','ne',OU['ne'],pNU,3.0,5.17,0.26)]:
    m = (codesA==cd) & np.isfinite(rA); dh = DEC[cd].predict(XA_[m]); dh1 = dh[:,0] if dh.ndim>1 else dh
    d1 = lagU(Dm_, 1)[m]; p1 = lagU(pim, 1)[m]; r = rA[m]; err = dh1-d1
    lb = lastbinA[m]; comp = pd.Series(dh1).groupby(lb).transform('mean').values   # in-sample bin means of decoded dose
    out = []
    for nm, dl in [('oracle nu', c*(d1-p1)), ('dec-pi', c*(dh1-p1)), ('dec-binmean', c*(dh1-comp))]:
        dE = (dl**2).mean() - 2*(r*dl).mean(); out.append('%s dNLL %.4f (x frac %.4f)' % (nm, dE/(2*sig**2), frac*dE/(2*sig**2)))
    print(cd, 'n', m.sum(), 'dec slope %.2f err sd %.3f corr(err,res) %.3f' % (np.polyfit(d1, dh1, 1)[0], err.std(), np.corrcoef(err, r)[0,1]), '|', '; '.join(out))
```