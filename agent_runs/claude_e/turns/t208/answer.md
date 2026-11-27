The tier steps explain only a little of the readout error. Its sd is 0.74 within 5 mg/dL of a threshold against about 0.6 elsewhere, roughly 14% of the error variance. Most of the error is in decoding the dose from the state, which a head edit cannot change. So the cost per unit of insulin gain is essentially fixed, and the practical path is:

1. Drop the NE part: it is costly, its cost swings strongly with how many NE rows a stay set has, it only helps NE ×½, and it slightly worsens NE ×2. Restore the original weights of units 131 and 159.
2. Spend the budget on the insulin gain.

I'll build `ins_only(a)`: `gated3(a, 0)` with units 131 and 159 fully restored. I'll then check offline on sel1 that it equals `gated3(a,0)` on glucose rows and touches nothing else. I'll also get the cost curve per row type for a = −2.2 to −2.8.

```python
ORIG_W0 = np.asarray(load_original(KMAP['W0']), np.float64); ORIG_b0 = np.asarray(load_original(KMAP['b0']), np.float64)
ORIG_W2 = np.asarray(load_original(KMAP['W2']), np.float64)
def ins_only(a):
    E = gated3(a, 0.0)
    E2 = {k: (v.clone() if isinstance(v, torch.Tensor) else np.array(v, copy=True)) for k, v in E.items()}
    for u in (131, 159):
        E2['W0'][u, :] = torch.as_tensor(ORIG_W0[u]) if isinstance(E2['W0'], torch.Tensor) else ORIG_W0[u]
        E2['b0'][u] = float(ORIG_b0[u])
        E2['W2'][:, u] = torch.as_tensor(ORIG_W2[:, u]) if isinstance(E2['W2'], torch.Tensor) else ORIG_W2[:, u]
    return E2
chg = {k: int((np.abs(np.asarray(ins_only(-2.4)[k], np.float64) - np.asarray(load_original(KMAP[k]), np.float64)) > 0).sum()) for k in KMAP}
print('changed entries ins_only:', chg)
for a_ in [-2.2, -2.4, -2.6, -2.8]:
    o_c = head_run(X, RR, ins_only(a_))
    R1['dn'] = np.asarray(o_c[1], float)[m1] - nll_orig1
    R1['dpm'] = np.asarray(o_c[0], float)[m1] - R1.pm_o
    g_ = R1.groupby(['code','drug']).dn.agg(['size','mean'])
    print(f'a={a_}', 'overall %.4f' % R1.dn.mean(), '|', ' '.join(f"{c}{'D' if d else '-'}:{m:+.4f}" for (c,d),(n,m) in g_.iterrows()),
          '| max|dpm| non-glu %.3g' % R1[~R1.code.isin([6,10])].dpm.abs().max())
```