The grep rules out a shortcut through `pending`. In the architecture, `pending_mlp` only feeds the action heads' filtered view. The reading head's query is `query_norm(view + type_emb)`, and its other inputs are features of the code being predicted. So the drug rate can reach the reading head only through `view = out + out_norm(mean h) + gap + z_head(z)`.

One term there has not been considered: `z_head(z)` adds the latent `z` directly to the view. If the rollout uses the posterior `z` (computed from the bundle, including the action token) at action bundles, a rank-1 edit of `z_post` plus `z_head` would be a far shorter channel than the prelude → b_in route. If the rollout uses the prior, that route is useless. Next I grep ARCHITECTURE.md for how `z` is computed and used, and when the prior or posterior applies.

```python
for i,l in enumerate(A.splitlines()):
    if re.search(r'z_post|z_prior|posterior|prior|z_head|z_in|latent|rollout|sampl|draw', l, re.I): print(i, l.strip()[:230])
```