The teacher-forced replica reproduces the stored states and the baseline NLL (1.40449), and one full pass over 100 stays takes only 0.5 s.

Killing low-spread dimensions is costly:

| Killed h dimensions | ΔNLL (all readings) | glucose_finger | glucose_lab |
|---|---|---|---|
| fast dims 1, 33, 17, 12 | +0.012 | +0.019 | +0.024 |
| 8 low-spread dims (incl. fast ones) | +0.013 | | |
| a different 8 (mostly slower dims) | +0.006 | | |

**Why the heuristic fails.** Small-spread, fast-decaying dimensions are working memory. They carry each bundle's fresh information into the rest of the state through `c_out`. A small spread does not mean a dimension is unimportant.

**Next step.** Scan all 192 dimensions one at a time with the same kill pattern (about 100 s in total) and rank them by NLL cost.

**What this decides.**
- *Viable:* there are enough cheap dimensions (cost near 0 each, combined cost well under 0.005) to hold 4–8 memory dimensions.
- *Not viable:* every dimension costs about 0.001 or more. Then I would cut the design to fewer dimensions, for example a single dimension per channel, or reconsider the approach.

```python
t0_ = time.time()
cost_all = np.zeros(192); cost_g = np.zeros(192); cost_m = np.zeros(192)
g6 = torch.tensor(codes_np == 6); g7 = torch.tensor(codes_np == 7)
for j in range(192):
    nllk = tf_nll(kill_dims(PO, [j]))
    d_ = nllk - nll0
    cost_all[j] = d_.mean().item(); cost_g[j] = d_[g6].mean().item(); cost_m[j] = d_[g7].mean().item()
print('scan time', round(time.time() - t0_, 1))
o_ = np.argsort(cost_all)
print('cheapest 24 dims (dim: all/glucose/map):')
print(' '.join(f"{j}:{cost_all[j]*1e3:.2f}/{cost_g[j]*1e3:.2f}/{cost_m[j]*1e3:.2f}" for j in o_[:24]))
print('quantiles of single-dim cost (x1e3):', np.round(np.quantile(cost_all, [0, .05, .1, .25, .5, .75, 1]) * 1e3, 3))
tau_np = tau_h.numpy()
print('tau_h of cheapest 24:', np.round(tau_np[o_[:24]], 2).tolist())
np.save('/work/dimcost.npy', np.stack([cost_all, cost_g, cost_m]))
```