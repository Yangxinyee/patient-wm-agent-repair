My offline forward reproduces the original NLL closely (1.5610 against 1.5557; the difference is the z-draws). On the held-out rows the current edit costs 0.0066 per reading. Almost all of it is on map_arterial (+0.023 per row), including +0.017 on rows with no NE.

That fits the old NE pair being gated only by code, not by "NE running", so its noisy readout is added on every arterial row. The transition table was in fact never changed, so the removed units' contributions are not compensated at all.

To see how much is the price of repurposing alone, I'll score gated3(0, 0), which keeps the four units repurposed but gives them zero output. Then I'll rank all 192 hidden units by how much removing each one (fixed at its mean) changes the factual NLL on training rows. If the current units are expensive and cheaper ones exist, switching units, plus least-squares compensation, will free budget.

```python
E00 = gated3(0.0, 0.0)
pm00, P00 = head_pm(E00); dn00 = head_nll(P00) - nll0
for nm, msk in [('all', np.ones(len(cs),bool)), ('gf', cs==6), ('gl', cs==10), ('ma', cs==7), ('mc', cs==5)]:
    print('removal-only %-4s dNLL/row %.4f  mean|shift| %.3f' % (nm, dn00[msk].mean(), np.abs(pm00-pm0)[msk].mean()))
Htr = gelu(np.einsum('nk,mk->nm', WCall[0].astype(np.float64), W0o) + ORIGd['b0'])
Hmean = Htr.mean(0)
contrib_sd = np.array([np.linalg.norm(W2o[:,u]) * Htr[:,u].std() for u in range(192)]) if 'W2o' in globals() else None
W2o = ORIGd['W2']
contrib_sd = np.array([np.linalg.norm(W2o[:,u]) * Htr[:,u].std() for u in range(192)])
order = np.argsort(contrib_sd)
print('current units contrib sd:', {u: round(float(contrib_sd[u]),4) for u in units})
print('20 smallest units:', [(int(u), round(float(contrib_sd[u]),4)) for u in order[:20]])
```