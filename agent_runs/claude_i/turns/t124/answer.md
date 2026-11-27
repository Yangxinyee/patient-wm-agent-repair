Found it. With the memory including the current bundle, ages in seconds, and the age embedding added to the memory vectors before both the key and value projections, the median one-step error is 0 for both h and out. So my core implementation matches the tool exactly on most bundles.

A few bundles still have errors of up to 11%. Next I will find which ones: their position in the stay, the gap since the previous bundle, the number of atoms, and whether they hold drug records. A systematic pattern would point to a remaining structural detail, such as how the age-zero entry or a particular bundle type is handled. I will also check how the first bundle of a stay is initialised (h = 0, out = 0?), so that I can run whole stays from scratch.

```python
V_ok = dict(incl=1, unit='s', age='both')
def step_err(k_, V=V_ok, hprev=None):
    hp_ = HHt[k_-1] if hprev is None else hprev
    hf = hp_ * torch.exp(-dts[k_]*torch.exp(WT['core.flow.a']))
    j0 = first_of[k_]; idx_m = np.arange(max(j0, k_+1-128), k_+1)
    ages = (tmin[k_] - tmin[idx_m]) * 60.
    hn, on = core_step(hf, Ee[k_], Uu[k_], ZZt[k_], Ee[idx_m], ages, V)
    return hn, on, ((hn-HHt[k_]).norm()/HHt[k_].norm()).item()
ER = np.full(NB, np.nan)
for k_ in range(1, NB):
    if prev_ok[k_]: ER[k_] = step_err(k_)[2]
bad_ = np.where(ER > 1e-3)[0]
print('checked', np.isfinite(ER).sum(), 'bad', len(bad_), 'median', np.nanmedian(ER), 'p99', np.nanpercentile(ER, 99))
pos_ = np.arange(NB) - first_of
print('bad positions', np.bincount(pos_[bad_])[:12], 'bad hours', np.round(H3[bad_][:15],2))
print('bad natoms', z3['n_atoms'][bad_][:15], 'has_ins', z3['has_insulin'][bad_][:15].astype(int), 'has_ne', z3['has_norepinephrine'][bad_][:15].astype(int))
# first bundle: try init h=0, z from tool
k0 = starts[0]
for init in ['zero']:
    hf0 = torch.zeros(4,192); hn, on = core_step(hf0, Ee[k0], Uu[k0], ZZt[k0], Ee[[k0]], torch.zeros(1), V_ok)
    print('first bundle err h', ((hn-HHt[k0]).norm()/HHt[k0].norm()).item(), 'out', ((on-OUTt[k0]).norm()/OUTt[k0].norm()).item())
```