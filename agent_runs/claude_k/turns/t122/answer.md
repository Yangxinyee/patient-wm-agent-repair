The drug atoms use small categorical tables (unit, rate unit, route, action, kind and event kind), and the `pending` vector is 192-dimensional and used only by the action heads. Rebuilding the action encoder exactly would mean searching these row conventions blind, because the tool never shows u_k directly. That is expensive.

First a cheaper consistency check. Right after the action bundle, `out` carried the action-B residual with R² 0.23–0.26. Yet my reconstructed q at the following reading gave only 0.04. Either the true head input loses that information (time flow, z draws, LayerNorm), or my q reconstruction is inaccurate.

I'll compute the same residual R² at the reading rows from the cached per-reading `out_`, `ctx_` and flowed slots. If out_/ctx_ keep R² around 0.2 while Q99 gives 0.04, the loss comes from my reconstruction or the z term, not the model. A value-head probe would then do better than it looked.

```python
pieces = {'Q99': Q99, 'out_': out_, 'ctx_': ctx_, 'hfmean': hf_.mean(1), 'WC99': WC99}
for nm, y in [('I1', dI[1]), ('N1', dN[1]), ('N2', dN[2])]:
    yu = y[ub]; yr = resid(yu, Cu)
    print(nm, ' '.join(f"{p}: {cvr2_(X[ub], yr, grp):.2f}" for p, X in pieces.items()))
print('corr out_ vs S2 prev-out check skipped; out_ std %.3f, ctx_ std %.3f, Q99 std %.3f' % (out_.std(), ctx_.std(), Q99.std()))
```