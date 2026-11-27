Gating on insulin presence is linearly separable with a clear margin (insulin bundles score ≥0.85, all others ≤−0.36). Two parts fail:
- **NE gate:** a single linear gate does not separate "NE present" from "absent"; the two score ranges overlap.
- **Rate readouts:** a single linear readout over all bundle types is inaccurate (insulin rms error 0.7 units/h or worse, NE rms 0.1–0.45). u_k is a strongly nonlinear function of the rate, and mixed insulin+NE bundles add a second dimension.

Before choosing a fix, I need three facts:
1. How common mixed insulin+NE bundles are in the records.
2. Whether readouts fitted separately for single-drug and mixed bundles are accurate.
3. Which bundles break the NE gate.

```python
ev_ = tr[tr.event.isin(['insulin','norepinephrine'])]
kb = ev_.groupby(['stay_id','hour']).event.agg(lambda s: tuple(sorted(s)))
cnt_ = kb.value_counts(); print({str(k): int(v) for k, v in cnt_.items()})
onlyI = hasI & ~hasN; bothm = hasI & hasN; onlyN = hasN & ~hasI
for nmz, msk in [('I only', onlyI), ('I+N', bothm)]:
    m_ = Ridge(alpha=1e-8).fit(GU[msk], GI[msk]); e_ = m_.predict(GU[msk]) - GI[msk]
    print(nmz, 'readout I: n %d rms %.4f max %.4f |w| %.0f' % (msk.sum(), np.sqrt((e_**2).mean()), np.abs(e_).max(), np.linalg.norm(m_.coef_)))
for nmz, msk in [('N only', onlyN), ('I+N', bothm)]:
    m_ = Ridge(alpha=1e-8).fit(GU[msk], GN[msk]); e_ = m_.predict(GU[msk]) - GN[msk]
    print(nmz, 'readout N: n %d rms %.4f max %.4f' % (msk.sum(), np.sqrt((e_**2).mean()), np.abs(e_).max()))
scN = GU @ wgN + bgN
viol = (hasN & (scN < maxnegN)) | (~hasN & (scN > minposN))
print('NE gate violators: hasN', hasN[viol].sum(), 'noN', (~hasN[viol]).sum(), ' rN of violators', np.round(np.unique(GN[viol & hasN])[:8],3), ' rI of noN violators', np.round(np.unique(GI[viol & ~hasN])[:8],2))
```