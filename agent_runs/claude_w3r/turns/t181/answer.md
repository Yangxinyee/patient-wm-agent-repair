`core_step_v` returns (y, out, E) and matches `step_at` exactly, so I have a validated step function for recursive local runs.

Next is the dedicated value direction. For each drug, fit ã with ã·û ≈ the start rate over the drug-bundle u tokens, with no intercept and constrained so the empty u gives exactly 0. I'll include synthetic bundles with rate and dose ×2 and ×½, so the channel stays linear over the edit range. I'm testing whether the residual sd is ≲ 0.05 for insulin and ≲ 0.02 for norepinephrine across factual and edited rates.

```python
import copy
def edit_atoms(atoms, code, mult):
    out_ = []
    for a in atoms:
        a = dict(a)
        if a['code']==code and a['kind']==0:
            if a.get('rate') is not None: a['rate'] = (a['rate'][0], val(a['rate'])*mult)
            if a.get('dose') is not None: a['dose'] = (a['dose'][0], val(a['dose'])*mult)
        out_.append(a)
    return out_
UX, YXi, YXn, GX = [], [], [], []
for s in sorted(recs.stay_id.unique()):
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        at = Bs[k][1]; ri = [val(a['rate']) for a in at if a['code']==ci and a['kind']==0]; rn = [val(a['rate']) for a in at if a['code']==cn and a['kind']==0]
        if not (ri or rn): continue
        for code_, mult in [(None,1),(ci,2),(ci,.5),(cn,2),(cn,.5)]:
            at2 = at if code_ is None else edit_atoms(at, code_, mult)
            if code_ is not None and not any(a['code']==code_ and a['kind']==0 for a in at): continue
            e2, u2 = enc_eu([a for a in at2])
            UX.append(np.asarray(u2, dtype=np.float64).ravel()); GX.append(int(s))
            YXi.append((ri[0] if ri else 0.0)*(mult if code_==ci else 1)); YXn.append((rn[0] if rn else 0.0)*(mult if code_==cn else 1))
UX, YXi, YXn, GX = map(np.array, (UX, YXi, YXn, GX)); print(UX.shape)
UXh = (UX - UX.mean(1, keepdims=True)) / np.sqrt(UX.var(1, keepdims=True) + 1e-5)      # x-hat of u token under norm1
chk_ = np.abs(UXh[:len(TOK['u'])] - TOK['u']).max() if False else None
tr_ = np.array([g not in test for g in GX])
def fit_lin(Xh, Y, tr, lam=1e-6):
    M_ = Xh[tr].T @ Xh[tr] + 1e4*np.outer(bg, bg) + lam*np.eye(192); return np.linalg.solve(M_, Xh[tr].T @ Y[tr])
for nm, Y in [('ins', YXi), ('ne', YXn)]:
    a_ = fit_lin(UXh, Y, tr_); r_ = Y[~tr_] - UXh[~tr_] @ a_
    print(nm, 'heldout resid sd', r_.std().round(4), 'max', np.abs(r_).max().round(3), '| by target>0', r_[Y[~tr_]>0].std().round(4), ' a.b1/g1', round(float(a_@bg),6))
```