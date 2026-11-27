Most of the z-clone checks pass:
- The capped insulin-only readout is exact (rms 0.001). The combined-pattern readouts are good to rms 0.055 (insulin) and 0.034 (NE).
- At non-drug bundles, fixing the z token at the mean measurement-bundle z token barely changes E: at most 0.011, mean 0.0001, against a spread of 0.11 across bundles. I will use that token.
- At drug bundles, the clone splits heads 1 to 3 exactly 50/50 between u and z (0.499/0.499), as designed. The mean change in E is only 0.00035.

But the largest single change is 0.83. My hypothesis: in some bundles or heads, u did not take nearly all the attention. There, adding a second token with u's key also takes weight from the slots.

The NE-presence gate is badly conditioned for a different reason. The u for (insulin 12, NE 0.59) lies 0.0095 from the u for (insulin 6.5 alone). That is because insulin saturates in the input encoding, and only the ×3 edit produces such combinations.

Next I look at the outliers:
- the distribution of each bundle's largest E change;
- how it relates to u's original attention weight in heads 1 to 3;
- what kind of bundles the outliers are.

If the outliers are bundles where u's weight was well below 1, the cure is to give the clone a lower attention logit than u, which shrinks the renormalisation effect. If instead they come from clone error, I need more components.

```python
import numpy as np, torch
dmax = (Pc['E'] - E_ref_d).abs().amax(dim=(1, 2)).numpy()
au = Pd['att'][:, 1:4, :4, 5].mean(2).numpy()
amin = au.min(1)
print('per-bundle max E change quantiles (50,90,99,99.9,max):', np.round(np.quantile(dmax, [.5, .9, .99, .999, 1.0]), 4))
print('fraction > 0.05:', (dmax > 0.05).mean().round(4), ' > 0.2:', (dmax > 0.2).mean().round(4))
print('orig u attention, min over heads 1-3 (quantiles 0.1%,1%,10%,50%):', np.round(np.quantile(amin, [.001, .01, .1, .5]), 4))
top = np.argsort(-dmax)[:8]
for i in top:
    print(f'bundle {i}: dmax {dmax[i]:.3f}  u-attn h1-3 {np.round(au[i], 3)}  ins {rId[i]} ne {rNd[i]}  hour {hrd[i]}  '
          f'clone resid {np.linalg.norm(proj[i].numpy() - Uk[i]):.4f}')
print('corr(dmax, 1-amin):', np.corrcoef(dmax, 1 - amin)[0, 1].round(3))
```