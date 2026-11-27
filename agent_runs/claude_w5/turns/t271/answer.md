The prelude attention at drug bundles is completely saturated. At j3, heads 0 and 3 put all their weight on the empty e token and heads 1 and 2 on u; at j5 and j7 every head attends only to u. So the attention output is just Wo applied to the e and u value vectors. Yet even a large common correction to it (27% of |ao|) only brings the j3 error to 5e-4. The original model's discrepancy isn't fully captured by any single simple correction I've tried, and the root cause stays unresolved.

I have about 50 runs left, so I'll stop chasing the exact cause. What matters for the edit is whether this small prelude discrepancy is what drives the tool's wildly different decoded x. The test: add the fitted u-token correction at j3 (norm 0.06) to my local edited model and check whether its implied injection coefficients move from local (−8.2, +5.1) toward the tool's (+29.3, −4.3).
- If they do, my decoders are hypersensitive to a tiny prelude difference. I'll redesign the edit to be insensitive to it and calibrate gains empirically with the tool's dose_edit readouts.
- If they don't, the edited-model gap has another cause.

```python
def msx_tokW(d, du_tok, W): return msx_tok(d, du_tok=None if du_tok is None else torch.tensor(du_tok), W=W)
dj = prep(3, Xo5)
s0_ = st_vec(msx_tok(dj, W=W64)); 
for nm, du_ in [('none', None), ('SOLU3', SOLU3), ('2xSOLU3', 2*SOLU3)]:
    so = st_vec(msx_tokW(dj, du_, W64)); se = st_vec(msx_tokW(dj, du_, WE))
    c, *_ = np.linalg.lstsq(A_, se-so, rcond=None)
    print(nm, 'local edit coef', c.round(3), ' orig-model err vs tool', (np.linalg.norm(Xo5[idx[3]]-so)/np.linalg.norm(so)).round(6), ' edited err vs tool', (np.linalg.norm(Xe5[idx[3]]-se)/np.linalg.norm(se)).round(6))
globals()['WCUR']=W64
```