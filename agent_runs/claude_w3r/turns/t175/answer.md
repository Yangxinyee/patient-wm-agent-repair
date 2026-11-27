Still a poor trade-off: drug-bundle u tokens and readings e tokens overlap in normalised space, so no linear readout gets both rate error and leakage small (0.11 vs 0.69, or 0.44 vs 0.02).

Better idea: stop head 2 from reading e at all. Head 1 already reads e at every bundle, and the readings also enter through cross-attention to memory. A rank-1 edit of head 2's key projection that lowers the e token's logit would leave head 2 reading only the empty u (a constant value) at non-drug bundles. A dedicated value direction then gives exactly zero leakage.

Next I'll test feasibility: whether a direction ρ separates e tokens from u tokens (ρ·(g1⊙ê) large for all e, ρ·(g1⊙û) ≈ 0 for drug u), and whether all head-2 slot queries share a common direction f.

```python
Xe_all = TOK['e']; Xu = TOK['u']
Ae, Au = Xe_all*g1n, Xu*g1n            # g1 ⊙ x-hat
from sklearn.svm import LinearSVC
Xc_ = np.vstack([Ae, Au]); yc_ = np.r_[np.ones(len(Ae)), np.zeros(len(Au))]
svm = LinearSVC(C=10, fit_intercept=False, max_iter=20000).fit(Xc_, 2*yc_-1)
rho = svm.coef_.ravel(); se, su = Ae @ rho, Au @ rho
print('rho.(g1*e) min/median', se.min().round(3), np.median(se).round(3), '  rho.(g1*u) max/median', su.max().round(3), np.median(su).round(3), ' rho.b1', round(float(rho@b1n_),4))
# head-2 queries at all bundles
Qs = []
for s in sorted(recs.stay_id.unique())[:40]:
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        A_, O_, T_, Tn_ = prelude_att(Bs, rs, EUs, k, ret_tokens=True)
        Qs.append((Tn_[:4] @ P['core.prelude.0.attn.q.weight'].T)[:, 96:144].numpy())
Qs = np.concatenate(Qs); fq = Qs.mean(0); fq /= np.linalg.norm(fq); pq = Qs @ fq
print('head2 query . mean-dir: min', pq.min().round(3), 'median', np.median(pq).round(3), ' |q| median', np.median(np.linalg.norm(Qs,axis=1)).round(3))
```