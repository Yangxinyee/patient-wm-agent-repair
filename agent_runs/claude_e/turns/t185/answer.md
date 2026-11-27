On held-out rows the current edit moves predictions noticeably where its corrections should be off. Mean |shift| is 0.35 on glucose rows without insulin, 0.50 on MAP-arterial rows without NE and 0.11 on cuff rows. The gates are clean, so this comes from removing the four repurposed units' original contributions, which the transition-table edit offsets only on average. That matches the tool's 0.53 shift and 0.0043 per-row cost on glucose rows without insulin, so a large share of the total cost is repurposing rather than the corrections.

Two fixes are possible: pick less important units, and offset the removed units' contribution by regressing it on the remaining hidden units and folding that into W2. Both are closed-form.

First I'll make my offline head forward score NLL, check it against RR.nll, and measure the current edit's held-out cost by code. That gives a baseline for comparing unit choices.

```python
obin = RR.obin.values.astype(int)
def head_nll(P): return -np.log(np.clip(P[np.arange(len(P)), obin], 1e-12, None))
nll0 = head_nll(P0); nllE = head_nll(PE)
print('orig NLL mine %.4f vs RR.nll %.4f' % (nll0.mean(), RR.nll.values.mean()))
dn = nllE - nll0
for nm, msk in [('all', np.ones(len(cs),bool)), ('gf', cs==6), ('gl', cs==10), ('ma', cs==7), ('mc', cs==5), ('gluc no ins', gl_rows & ~hi_s), ('ma no NE', (cs==7)&~pv_ne)]:
    print('%-12s n %5d dNLL/row %.4f' % (nm, msk.sum(), dn[msk].mean()))
for s_ in ['1','A']:
    print('src', s_, 'overall dNLL %.4f' % dn[src==s_].mean())
dTR = (Ecur['TR'].numpy() if hasattr(Ecur['TR'],'numpy') else np.asarray(Ecur['TR'])) - ORIGd['TR']
print('TR edit: nonzero entries', int((np.abs(dTR)>1e-9).sum()), 'max', np.abs(dTR).max().round(3))
```