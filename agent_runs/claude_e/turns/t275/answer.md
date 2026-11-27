`head_run` returns a tuple of four arrays: predicted mean per row, NLL per row, the 17 bin probabilities, and the hidden activations (16 z-draws × rows × 192 units). The hidden activations are exactly what the compensation regression needs.

In this run I will:
1. Confirm the first two outputs match the tool's recorded predicted means and NLLs (columns `pm` and `nll`).
2. Check whether `head_run` honours an output-bias entry `b2`. If it does, compensation can use the bias.
3. Measure offline the per-event cost on all sel1 rows of `ins_only(-2.7)` and of `gated3(-2.7, a_ne)` for a_ne = 0, 1, 2, 3.

This separates the NE part's repurposing cost (a_ne=0) from its gain-dependent cost. The sel1 rows are drug-heavy, so the absolute numbers will be higher than on random stays; the earlier ratio of tool to sel1 cost was about 0.4–0.5.

```python
pm_o, nll_o, P_o, H_o = head_run(X, R1, orig)
print('check pm corr', np.corrcoef(pm_o, R1['pm'].values)[0,1], 'nll corr', np.corrcoef(nll_o, R1['nll'].values)[0,1],
      'mean nll off/tool', nll_o.mean(), R1['nll'].mean())
ob = dict(orig); ob['b2'] = load_original('attr_heads.value_bin.2.bias').double().clone(); ob['b2'][5] += 5.0
print('b2 honoured?', not np.allclose(head_run(X, R1.iloc[:300], ob)[1], nll_o[:300]))
ev = R1['event'].values
def cost(Wd, lab):
    _, nl, _, _ = head_run(X, R1, Wd); d = nl - nll_o
    s = {e: round(d[ev==e].mean(),4) for e in np.unique(ev)}
    print(f'{lab:>16} all {d.mean():.4f}', s); return d
d_i = cost(gi, 'ins_only-2.7')
D = {}
for a in [0.0, 1.0, 2.0, 3.0]:
    D[a] = cost(gated3(-2.7, a), f'gated3 -2.7/{a}')
print('NE part extra (all rows):', {a: round((D[a]-d_i).mean(),4) for a in D})
```