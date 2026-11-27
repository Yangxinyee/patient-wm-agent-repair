`F` columns are [hour, previous value, current value, dose lag 1, lag 2, lag 3], `Lp_` repeats the three dose lags, and `cp_` holds the code names. So `V` (with_code from the perturbed-dose samples) plus `F[:,3]` is a training set for a lag-1 dose decoder. Next I'll fit ridge decoders for insulin on the finger rows, both plain and within-cell (centred by previous-value bins), and score each on three things:
- **Edit response:** mean response to the insulin ×2 and ×½ edits, which should be close to the actual dose change.
- **Factual error:** within-cell sd of (decoded minus true lag-1 dose) on the factual rows, which should be under about 0.5 u/h.
- **Leakage:** response to the NE edits.

```python
fm = (cp_ == 'glucose_finger'); print("codes", np.unique(cp_), fm.sum())
Vg, yg = V[fm], F[fm, 3]; cg = np.round(F[fm, 1] / 10)
def cres(A_, c_):
    A_ = A_.copy(); 
    for u_ in np.unique(c_):
        k_ = c_ == u_; A_[k_] -= A_[k_].mean(0)
    return A_
fac_m = cf == 0
cellf = lbf[fac_m]
def score(w_, b_):
    out = []
    for (dr, fac), dct in EDR.items():
        gi_ = dct['gi']; rc_ = dct['reached'].astype(bool) & (cf[gi_] == 0)
        r_ = dct['dX'][rc_] @ w_; out.append(f"{dr[:3]}x{fac} {r_.mean():.2f}/{dct['dd'][rc_,0].mean():.2f}")
    e_ = Xf[fac_m] @ w_ + b_ - dI1[fac_m]
    e_c = e_ - np.array([e_[cellf == u_].mean() for u_ in cellf]) if False else cres(e_[:, None], cellf)[:, 0]
    return " | ".join(out) + f" | fact err sd {e_.std():.3f} within-cell {e_c.std():.3f}"
for mode_ in ['plain', 'within']:
    A_ = Vg if mode_ == 'plain' else cres(Vg, cg); t_y = yg if mode_ == 'plain' else cres(yg[:, None], cg)[:, 0]
    mA_, my_ = A_.mean(0), t_y.mean(); S_ = (A_ - mA_).T @ (A_ - mA_); c_ = (A_ - mA_).T @ (t_y - my_)
    for lam in [1e1, 1e2, 1e3]:
        w_ = np.linalg.solve(S_ + lam * np.eye(192), c_); b_ = (yg - Vg @ w_).mean()
        print(mode_, lam, score(w_, b_))
```