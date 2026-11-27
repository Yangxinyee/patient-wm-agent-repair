Using the recorded value ('obs') as the bin target, with the z-prior's second half as a log-std, reproduces the tool. The mean is 1.4342 against the tool's 1.4364, with per-row correlation 0.963; the remaining difference is z-sampling noise. So my earlier ΔNLL used the wrong targets.

Next I redo the edited-versus-original comparison with this setup, averaging 4 seeds of common random numbers, overall and by reading type. I also redo the a/b decomposition (first-order benefit versus quadratic cost of the shift). If the corrected ΔNLL is clearly below 0.005 and improves with shifts that point the right way, the current design is fine and I move on to rollout calibration. If not, the direction needs work.

```python
def nll_seeds(V, seeds=(0, 1, 2, 3)):
    return np.mean([nll_var(V, tgt_obs, 'logstd', nz=4, seed=s_) for s_ in seeds], axis=0)
n0o = nll_seeds(VO); n1o = nll_seeds(VEd)
print('obs-bin NLL orig', n0o.mean().round(4), 'edited', n1o.mean().round(4), 'diff', (n1o-n0o).mean().round(5))
for c in np.unique(codes_):
    m = codes_ == c; print('  ', c, m.sum(), 'diff', (n1o[m]-n0o[m]).mean().round(5))
for c in ('glucose_finger', 'map_arterial'):
    m = codes_[ia] == c; e_ = err0[ia][m]; d_ = dm[ia][m]; dn = (n1o-n0o)[ia][m]
    co = np.linalg.lstsq(np.c_[e_*d_, d_**2], dn, rcond=None)[0]
    dd = np.abs(d_); qs_ = np.quantile(dd, [0, .25, .5, .75, 1.])
    print(c, 'a', (-co[0]).round(5), 'b', co[1].round(5), '| dNLL by |shift| quartile', [round(dn[(dd>=qs_[i])&(dd<=qs_[i+1])].mean(),5) for i in range(4)])
```