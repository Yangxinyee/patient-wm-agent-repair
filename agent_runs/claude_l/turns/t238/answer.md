The rollout replica builds z from `W['z_post…']` and `W['z_in…']` inside `bstep`. So any weight edit I put in W runs straight through the offline rollout, and through teacher forcing, with no extra hooks. That is the testbed I'll use.

My plan stays the one in my notes:
- z_post computes, from `[ctx, e, u]`, a low-rank copy of u plus a scaled dose residual r = rate − π̂(ctx).
- z_in turns that into a z token equal to u + gain·r·v.
- Heads 0–2, which attend hard to the u token, then split between two near-identical tokens. With r = 0 behaviour is unchanged; with r ≠ 0 the residual dose acts on the next reading.

To build it I need, on real bundles, the ctx, the u, and the exact rates. Arrays named DA_ctx, DA_u and DA_meta (57,239 bundles) look like exactly that collection, but I don't remember the meta columns. I'll print compact per-column summaries for DA_meta and DM, and check whether DA_ctx/DC and DA_u/DU are the same arrays. If the columns turn out to be stay, hour, has-insulin, insulin rate, has-NE and NE rate, I can fit the decoders next run.

```python
import numpy as np
for nm in ['DA_meta', 'DM']:
    A_ = np.asarray(globals()[nm])
    print(nm, A_.shape)
    for j in range(A_.shape[1]):
        c_ = A_[:, j]
        print(f"  col{j}: min {c_.min():.3f} max {c_.max():.3f} mean {c_.mean():.3f} nuniq {len(np.unique(c_))} nz {np.mean(c_!=0):.3f}")
print('HR', np.min(HR), np.max(HR), len(np.unique(HR)))
print('DA_ctx==DC', np.allclose(DA_ctx, DC), 'DA_u==DU', np.allclose(DA_u, DU))
print('first rows meta', np.round(np.asarray(DA_meta)[:4], 3).tolist())
```