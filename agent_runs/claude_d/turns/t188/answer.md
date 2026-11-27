The insulin direction can be gated well: with λ = 1000 it gives about 0.8 mg/dL per unit and only 0.12 leakage into MAP at ±5. The NE direction cannot. Every setting either barely moves MAP (0.3 per unit) or leaks several mg/dL into glucose. The shared hidden layer seems unable to shift MAP means cleanly through a single pre-activation direction.

A cleaner design is to repurpose a pair of hidden units as a gated linear channel:

- Both units get pre-activation +M × (MAP-vs-glucose separator), so they sit deep in GELU's linear regime on MAP rows and are silenced on glucose rows.
- One unit also gets the decoded NE signal.
- Their output columns get +t and -t, where t is a logit tilt over MAP bins. The separator's noise then cancels exactly, and MAP logits move by exactly k·t·s.

This only works if some hidden units are nearly worthless for the factual fit. So next I measure, for every one of the 192 hidden units, how much offline NLL rises when the unit is replaced by its mean activation.

```python
Hh = gelu(A); lg0 = Hh@W2.T + b2 + HI['trans']
def nll_of(lg):
    lg = lg - lg.max(1,keepdims=True); p = np.exp(lg); p/=p.sum(1,keepdims=True)
    return -np.log(p[np.arange(len(p)), ob]+1e-300)
base_nll = nll_of(lg0).mean()
mh = Hh.mean(0); cost = np.zeros(192)
for j in range(192):
    lg = lg0 - np.outer(Hh[:,j]-mh[j], W2[:,j])
    cost[j] = nll_of(lg).mean() - base_nll
o = np.argsort(cost)
print('base', round(float(base_nll),4), 'cheapest', [(int(j), round(float(cost[j]),5), round(float(Hh[:,j].std()),3)) for j in o[:8]])
print('median cost', round(float(np.median(cost)),4))
```