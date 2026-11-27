Removing the action token moves glucose predictions by only about 0.02–0.16 per unit rate at lags 1–6, versus the true effect of about −2. So the token's whole influence on the state is roughly 1/13 of what is needed, not just its rate dependence. Making its content depend more on the rate cannot fix that, which explains why input scaling saturated. The token needs more weight in the prelude's attention.

I can't see token contents, but I can steer what the encoder emits. A rank-1 update on `encode_action.norm.bias` can make it point along any direction I compute. If the action token looked like the slot tokens, which are the layer-normalised state h, the slots might attend to it more. That only works if h's direction is stable across bundles, so I'll check that in an existing states file first.

```python
import glob, os
fs = sorted(glob.glob('/work/fwd_*/states_*.npz'))
print([f.split('/')[-2]+'/'+os.path.basename(f) for f in fs])
S = np.load(fs[0], allow_pickle=True); print({k: S[k].shape for k in S.files})
```