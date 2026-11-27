Untrained rows have norm about 0.277, which is the init norm for std 0.02, and row 0 is padding wherever its norm is 0. That tells me which embedding rows are actually used:
- **Categorical:** unit rows 2–5 (four units), rate_unit rows 2–3, and the trained kind/ekind rows. Status, route, action, text and discharged_to are effectively unused.
- **kq:** only row 0 is trained, so it is probably a constant added to every atom.
- **type_emb:** 15 entries, likely 5 event kinds × 3 atom kinds, used as the query type in the heads.
- **Time flow:** timescales of 573 s to 1.9 years, so dt is in seconds.

The first bundle of every stay is the same lone hospital-admission atom at −2 h with no history. If its states are identical across stays, it is a fixed computation I can match exactly to pin down the embedding, activation, z-posterior split and initial state.

I'll check that, and look at the magnitudes of out, the h slots and the z posterior mean.

```python
first_idx = [np.where(sid_==s)[0][0] for s in selT]
F = X[first_idx]
print("max abs diff across stays (first bundle):", np.abs(F - F[0]).max())
print("hours of first bundle:", np.unique(hr_[first_idx]), " n_atoms:", np.unique(nat[first_idx]))
x0 = F[0]
print("out[:8]", x0[:8].round(4), " out norm", np.linalg.norm(x0[:192]).round(3), " mean/std", x0[:192].mean().round(4), x0[:192].std().round(4))
for s in range(4):
    hs = x0[192+192*s:384+192*s]; print("slot", s, "norm", np.linalg.norm(hs).round(3), hs[:5].round(4))
print("z post mean:", x0[960:].round(3))
# second bundle: hour -1, identical?
sec_idx = [np.where(sid_==s)[0][1] for s in selT]
print("second bundle hours:", np.unique(hr_[sec_idx]))
```