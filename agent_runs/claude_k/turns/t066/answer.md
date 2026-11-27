The posterior rebuild has stalled. Correlation is about 0.85 and the error about 0.045, and none of the remaining encoder details moves it. I will accept this approximate precision rather than spend more runs chasing an exact match.

That matters for strategy. An exact numpy copy of the core does not look reachable, so I need an edit whose effect I can measure with forward passes rather than fully compute.

The key fact so far is that q, the only thing the value head sees, does not carry the random part of the dose. The heads see the state only through the mean over the four slots. So the dose may still be present in the slot-specific part of h, or in the latent z, which `z_post` computes directly from u_k.

This run probes the full saved state right after each insulin bundle for the random part of that bundle's dose, using stay-grouped ridge regression. It compares five feature sets: out alone, the slot mean, all four slots, z alone, and everything together.

- **High R² from the slots or z but not from out:** the information exists and is lost in the summary. Moving it into out/q is then a targeted edit.
- **Low R² everywhere:** the dose barely enters the state, and the edit must start at the action encoder.

```python
ia = st.index[(st.hi.values) & (st.hour.values >= 0)].values
sa = st.stay.values[ia]; ta = np.floor(st.hour.values[ia]).astype(int)
ra = np.array([sidx[s_] for s_ in sa])
dA = ins_all[ra, ta]
lgA = glast[ra, ta]; onA = np.where(ta >= 1, ins_all[ra, np.maximum(ta - 1, 0)] > 0, False).astype(float)
bA = np.digitize(np.nan_to_num(lgA, nan=150), [130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 300])
ZA = np.column_stack([np.eye(12)[bA], np.eye(12)[bA] * onA[:, None]])
cA, *_ = np.linalg.lstsq(ZA, dA, rcond=None); dresA = dA - ZA @ cA
print('n insulin bundles', len(ia), 'dose sd %.2f resid sd %.2f' % (dA.std(), dresA.std()))
XA = X[ia].astype(np.float64)
feats = {'out': XA[:, :192], 'slotmean': XA[:, 192:960].reshape(-1, 4, 192).mean(1), 'h (4 slots)': XA[:, 192:960], 'z': XA[:, 960:992], 'all': XA}
for nm_, F_ in feats.items():
    for tn, y_ in [('dose', dA), ('dose resid', dresA)]:
        yp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), F_, y_, groups=sa, cv=GroupKFold(5))
        print('%-12s %-11s CV R2 %.3f' % (nm_, tn, 1 - np.mean((y_ - yp) ** 2) / np.var(y_)))
```