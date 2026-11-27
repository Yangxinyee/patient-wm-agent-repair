Run 75 gives me the full head structure:
- `query_norm` is a LayerNorm, followed by `type_emb` and `embed.code`.
- `last_numeric` is a linear map from 8 inputs.
- `attr_heads.value_bin` is an MLP from 192 to 192 to 17.
- `embed_transition.value_bin` is a 221×17 table: 13 codes × 17 last-value bins.
- `z_prior` and `z_head` handle the latent, and `config.z_eval_samples` is 4.

This confirms that the head averages over 4 prior draws of z. I will add that to the head re-implementation later.

**The deciding question is propagation, not the head.** Suppose I correct only the lag-1 and lag-2 dose terms in the value head. Then the 12-hour accumulation in the rollout is set by how the model carries its own drawn glucose forward. The original model's rollout shift grows almost linearly with lag. That hints the model carries a glucose shift almost fully from hour to hour (ρ̃ ≈ 1). My impulse response decays about 0.8 per hour from lag 3 on. That fits two different explanations:
- (b) glucose itself mean-reverts with φ ≈ 0.8 and insulin acts only briefly;
- (a) glucose behaves like a random walk and insulin's effect partly reverses later.

These need different corrections, so I must settle which one the records support.

**Plan.**
1. Locally on the train grid, regress glucose_finger at t on its own lags and on dose lags. This gives the records' one-step coefficients.
2. Later, regress the model's teacher-forced predicted means on the same features, using a few factual requests on train stays. This gives the model's coefficients.
3. Compare the two sets of coefficients.

First I need to recall what `gg` and the earlier forward outputs contain, so I list their columns and files.

```python
import os, pandas as pd
print(list(gg.columns))
print(gg.head(3).T.head(96).to_string()[:3500])
for d_ in sorted(os.listdir('/work')):
    p_ = os.path.join('/work', d_)
    if os.path.isdir(p_): print(d_, os.listdir(p_))
print(open('/data/README.md').read()[:1500])
```