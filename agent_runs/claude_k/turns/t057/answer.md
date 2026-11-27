My posterior means barely correlate with the saved ones: most dimensions sit between −0.3 and +0.3. The prior computed from the same context correlates 0.4–0.7. The error also grows with the number of atoms in a bundle.

So the context is roughly right, and the problem is in how a bundle's own content enters. One possibility is an alignment problem: the saved z might belong to the bundle before or after the one I am pairing it with.

This run compares my posterior and prior with the saved z shifted by −1, 0 and +1 bundles within each stay. If a shifted alignment gives much higher correlation, the saved states are offset from my indexing. If not, the problem is in my measurement encoding e_k itself.

```python
stay_of = st.stay.values
def corr_shift(A_, shift):
    idxs = np.array([c_[0] for c_ in cand2]) + shift
    ok = np.array([0 <= i_ < len(st) and stay_of[i_] == stay_of[c_[0]] for i_, c_ in zip(idxs, cand2)])
    T_ = X[idxs[ok], 960:992]
    return np.mean([np.corrcoef(A_[ok][:, j], T_[:, j])[0, 1] for j in range(32)])
for sh in [-2, -1, 0, 1, 2]:
    print('shift', sh, 'post corr %.3f' % corr_shift(ZP, sh), 'prior corr %.3f' % corr_shift(ZPR, sh))
idx_, ip, dt, g_ = cand2[0]
print(st.loc[ip - 1:idx_ + 1]); print(g_[['hour', 'event', 'value']])
```