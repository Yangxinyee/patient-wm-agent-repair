My reproduction is now exact for 87% of bundles (median error 2e-7), and the first bundle of a stay starts from h = 0 exactly.

The 672 failures follow a clear pattern. They are bundles that hold both insulin and norepinephrine records, plus the final bundle at hour 25.17, which has two atoms and is probably the two END records. So the problem is how my encoder builds e or u when a bundle holds atoms of both drugs, not the core.

Next I will look at the z error from my earlier z check (`erF`) for these bundles:
- **If z is also wrong there:** the cause is the e or u encoding. The likely candidates are how END atoms of two drugs are combined, or which atoms count as actions.
- **If z is right but h is wrong:** the core handles these bundles differently.

I will also test whether recomputing e with only one END atom, or another simple variant, removes the error.

```python
erF_ = np.asarray(erF)
print('erF shape', erF_.shape, 'okm on bad', okm[bad_].mean(), 'okm on good', okm[np.setdiff1d(np.where(np.isfinite(ER))[0], bad_)].mean())
eb = erF_[bad_] if erF_.shape[0] == NB else None
if eb is not None: print('z err on bad: median', np.median(eb), 'max', eb.max())
# bundle kinds: both-drug bundles vs end-of-stay
both_ = (z3['has_insulin'][bad_]==1) & (z3['has_norepinephrine'][bad_]==1)
print('bad both-drug', both_.sum(), 'bad other', (~both_).sum(), 'hours of other', np.unique(np.round(H3[bad_][~both_],2))[:10])
allboth = np.where((z3['has_insulin']==1)&(z3['has_norepinephrine']==1))[0]
print('all both-drug bundles', len(allboth), 'of which bad', np.isin(allboth, bad_).sum())
```