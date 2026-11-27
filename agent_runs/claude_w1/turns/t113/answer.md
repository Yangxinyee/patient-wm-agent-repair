The vector is genuinely history-dependent. An early reading change propagates and decays over later rows, and the doubled dose produces a small change. So `gen_samples` is evidently a fast batched version of my replica's head input, and I can generate large sample sets cheaply.

That makes a direct feasibility test possible, run separately per drug and to be done next:
1. Pair factual records with copies whose insulin doses are randomly rescaled.
2. Regress the paired head-input differences on the dose changes at each lag. This gives the exact "dose directions" in head-input space.
3. Compute the minimum-variance decoder with fixed response to those directions, and measure its error on factual states.

Before that, I need the insulin policy expectation E[dose | glucose used, previous on/off], including what happens in hours with no glucose reading. That expectation is the reference against which innovations are measured.

```python
p2 = Ptr2
print(p2[['Gany','D','prev_on','gb']].describe().loc[['count','mean','min','max']].round(2).to_string())
nr = p2.Gany.isna()
Dprev = p2.groupby(level=0).D.shift(1)
print('hours w/o glucose reading: %d, of which D>0: %d' % (nr.sum(), (nr & (p2.D > 0)).sum()))
sub = p2[nr & Dprev.notna()]
print('no-reading hours: corr(D, Dprev)=%.3f, frac D==Dprev %.3f, mean D %.2f vs Dprev %.2f' % (np.corrcoef(sub.D.fillna(0), Dprev[sub.index].fillna(0))[0,1], (sub.D.fillna(0)==Dprev[sub.index].fillna(0)).mean(), sub.D.fillna(0).mean(), Dprev[sub.index].fillna(0).mean()))
```