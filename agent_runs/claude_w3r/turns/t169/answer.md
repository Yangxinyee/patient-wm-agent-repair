Fitting with zero targets at non-drug bundles fails. Residual sd is 0.75 at insulin bundles and 0.21 at non-drug bundles (max 2.5), so head 2's output cannot be both rate-coding and zero elsewhere. I also confirmed the 4 slots get identical head-2 outputs.

A better gate is a dedicated value direction: a vector a with a·norm1(token) ≈ rate for the u token and ≈ 0 for slot, e and z tokens, and a·b1 = 0 so that u = 0 gives exactly 0. That value would replace head 2's least-used value direction (smallest singular direction of its output projection), giving a channel that is exactly zero whenever head 2 doesn't look at a drug-bearing u. Next I'll test feasibility: a weighted least-squares fit of a on the normalised tokens, with held-out residuals.

```python
TOK = {'u': [], 'slot': [], 'e': [], 'z': []}; Yi, Yn, Gu, Go = [], [], [], []
for s in sorted(recs.stay_id.unique()):
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        A_, O_, T_, Tn_ = prelude_att(Bs, rs, EUs, k, ret_tokens=True)
        xh = ((Tn_ - P['core.prelude.0.norm1.bias']) / P['core.prelude.0.norm1.weight']).numpy()   # standardized tokens x-hat
        at = Bs[k][1]; ri = [val(a['rate']) for a in at if a['code']==ci and a['kind']==0]; rn = [val(a['rate']) for a in at if a['code']==cn and a['kind']==0]
        if ri or rn: TOK['u'].append(xh[5]); Yi.append(ri[0] if ri else 0.0); Yn.append(rn[0] if rn else 0.0); Gu.append(int(s))
        TOK['slot'].append(xh[0]); TOK['e'].append(xh[4]); TOK['z'].append(xh[6]); Go.append(int(s))
TOK = {k: np.array(v) for k, v in TOK.items()}; Yi, Yn, Gu, Go = map(np.array, (Yi, Yn, Gu, Go))
Xoth = np.concatenate([TOK['slot'], TOK['e'], TOK['z']]); Goth = np.concatenate([Go, Go, Go])
print({k: v.shape for k, v in TOK.items()})
def fit_a(Y, wt, tr_u, tr_o, lam=1e-4):
    A_ = TOK['u'][tr_u]; O_ = Xoth[tr_o]
    M_ = A_.T @ A_ + wt * O_.T @ O_ + lam*np.eye(192); return np.linalg.solve(M_, A_.T @ Y[tr_u])
test = set(sorted(np.unique(Go))[::5])
tu = np.array([g not in test for g in Gu]); to = np.array([g not in test for g in Goth])
for wt in [0.1, 1, 10]:
    for nm, Y in [('ins', Yi), ('ne', Yn)]:
        a_ = fit_a(Y, wt, tu, to); ru = Y[~tu] - TOK['u'][~tu] @ a_; ro = Xoth[~to] @ a_
        print(wt, nm, 'u resid sd', ru.std().round(4), 'others sd', ro.std().round(4), 'max', np.abs(ro).max().round(3))
```