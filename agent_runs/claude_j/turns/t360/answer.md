Forward pass 19 measured the fallback mk2(2.2, 5) on batch 3 at −0.0003, a small improvement in fit.

| Stay set | Cost of mk2(2.2, 5) |
|---|---|
| Batch 1 | +0.0018 |
| Batch 2 | +0.0010 |
| Batch 3 | −0.0003 |
| All 300 stays, weighted by readings | about +0.0008 |

So the fallback is comfortably safe. The cost varies between 100-stay sets by roughly ±0.001.

**Two patterns from all the runs:**
- The cost grows faster than linearly with edit size: batch 1 went from +0.0018 here to +0.0100 at mk2(4.5, 15).
- The B channel's M-event cost is close to quadratic, so it is cheap when moderate. The A channel's G-event cost is closer to linear.

**Final candidate (Python run 299 of 300):** mk2(2.5, 8). It spends a bit more on A, which drives the main readout, and more on B, which buys the most improvement per unit cost.
- My fits predict about +0.0029 on batch 1, the costliest batch, and about +0.0013 averaged over the population.
- Predicted effects: A x2 ≈ −3.5, A x1/2 ≈ +1.5, B x2 ≈ +1.4, B x1/2 ≈ −1.1. The originals were −2.65, +1.27, +0.75 and −0.44.

**How I will validate it:** I will check it on all four non-test batches and the test set, then measure its effects with dose_edit.
- It is kept if no batch exceeds about 0.0035 and the reading-weighted mean stays at or below about 0.002.
- Otherwise run 300 reverts to mk2(2.2, 5).

```python
import numpy as np
mk2(2.5,8.0)
W1=np.asarray(load('core.prelude.0.attn.kv.weight'),dtype=np.float64); print('saved |D| %.3f'%np.linalg.norm(W1-W0), edited())
with open('/work/notes.md','a') as fh: fh.write("""- FALLBACK mk2(2.2,5) factual: b1 +.00179 (GF+.003 GL+.0052 MA+.0014 MC0), b2 +.00096, b3 -.00027 -> pop ~+.0008. (run298 split analysis failed: event col not found).
- b1 cost super-linear: G~.00341lA+.00213lA^2 (lA=(kA-1)/3.5), M~-.00017lB+.00463lB^2 (lB=(kB-1)/14).
- Run299: FINAL candidate mk2(2.5,8): pred b1 ~.0029, pop ~.0013; pred A2 -3.5 Ah 1.5 B2 1.4 Bh -1.1 (orig -2.65 1.27 .745 -.435; truth sel1 -13.95 6.98 4.28 -2.14).
  Validate b1-b4 + sel1 + dose_edit sel1. If any batch >.0035 or mean >.002 -> run300 = mk2(2.2,5) revert (last run!).
""")
```