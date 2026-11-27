Per dimension, my computed posterior tracks the true z only partly. The high-variance dimensions match well (dims 4 and 5: correlation 0.92 and 0.91, slopes near 1), but many low-variance dimensions do not (median correlation 0.43).

Two other results narrow it down:
- Zeroing e doubles the error, so e matters and is roughly right.
- Precision-weighting with the prior makes things worse, so the posterior is the direct `z_post` output.

That pattern fits a partly scrambled input to `z_post`. The leading suspect is the order of the three 192-wide blocks in the 576-wide input. The documentation says `[ctx, e_k, u_k]`, but my notes from the earlier session record "ecu" as a variant I tried.

Test: evaluate all 6 block orders, re-searching the activations. If one order gives an error near zero, the posterior input is settled. If all stay around 0.05, the problem lies in the encoder.

```python
blocks_ = {'c': ctx_b, 'e': E1g, 'u': torch.zeros_like(E1g)}
res_perm = []
for perm in itertools.permutations('ceu'):
    inp_ = torch.cat([blocks_[p_] for p_ in perm], 1)
    for zn in ['relu', 'gelu', 'silu', 'tanh']:
        zz_ = mlp2l(inp_, 'z_post', ACT[zn])
        for half in [0, 1]:
            er = (zz_[:, 32*half:32*half+32] - z1_true).abs()
            res_perm.append((er.mean().item(), er.max().item(), ''.join(perm), zn, half))
res_perm.sort(key=lambda r: r[0])
for r_ in res_perm[:8]: print(np.round(r_[:2], 5), r_[2:])
# also: does any ctx alone (no e) explain z for the u-only (drug) bundles? quick count of bundle types overall
types_ = pd.Series([('act' if any(a_['act'] for a_ in bl[1]) else '') + ('meas' if any(not a_['act'] for a_ in bl[1]) else '') for s_ in batch1 for bl in BUN[s_]])
print(types_.value_counts())
```