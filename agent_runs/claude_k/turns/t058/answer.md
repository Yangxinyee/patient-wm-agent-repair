Shifting the alignment does not help. My posterior means stay uncorrelated with the saved ones at every shift, while the prior keeps a correlation of about 0.5 at shift 0. So the pairing of bundles is right and the context is roughly right. The way I build the bundle encoding, or how it enters `z_post`, is wrong.

This run screens structural variants using mean per-dimension correlation, which judges direction better than absolute error:
- all orderings of the three `z_post` inputs;
- which half of the output is the mean;
- the set encoder with or without its final norm and size term;
- averaging instead of dividing by √n;
- atoms layer-normed before `mlp1`.

A variant that lifts the correlation well above the prior's 0.5 identifies the right structure. If none does, the saved z is probably not a simple posterior mean of this form, and I will drop this check in favour of validating whole state updates.

```python
import itertools as it
def enc_variant(vecs, p, mode):
    V_ = np.array(vecs)
    if 'atomln' in mode: V_ = ln(V_, np.ones(192), np.zeros(192))
    M1 = mlp(V_, p + '.mlp1'); n_ = len(V_)
    s_ = M1.mean(0) if 'mean' in mode else M1.sum(0) / np.sqrt(n_)
    if 'nosize' not in mode: s_ = s_ + lin(np.array([np.log1p(n_)]), p + '.size')
    y_ = mlp(s_, p + '.mlp2')
    return y_ if 'nonorm' in mode else ln(y_, W[p + '.norm.weight'], W[p + '.norm.bias'])
pre_c = [(ctx_of(ip, dt), [base_mk(e_, v_) for e_, v_ in zip(g_.event, g_.value)], X[idx_, 960:992]) for idx_, ip, dt, g_ in cand2]
def score(mode, order, half):
    P_, T_ = [], []
    for c_, vecs, t_ in pre_c:
        e_k = enc_variant(vecs, 'encode', mode); parts = {'c': c_, 'e': e_k, 'u': np.zeros(192)}
        o_ = mlp(np.concatenate([parts[k_] for k_ in order]), 'z_post'); P_.append(o_[:32] if half == 0 else o_[32:]); T_.append(t_)
    P_, T_ = np.array(P_), np.array(T_)
    return np.nanmean([np.corrcoef(P_[:, j], T_[:, j])[0, 1] for j in range(32)])
out_s = []
for mode in ['', 'mean', 'nosize', 'nonorm', 'atomln']:
    for order in it.permutations('ceu'):
        for half in [0, 1]:
            out_s.append((score(mode, order, half), mode, ''.join(order), half))
out_s.sort(reverse=True)
for o_ in out_s[:8]: print(o_)
```