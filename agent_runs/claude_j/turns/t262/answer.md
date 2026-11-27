Run 234 shows that `step`'s output does not depend on z at all; the difference is about 1e-15. Since the replica matched the tool's states, the real model probably has (near-)zero weights where z enters the state, a collapsed latent. If so, the z channel is unused. Its input weights (`core.z_in`) would then be free to repurpose as a clean path for a dose signal into the prelude.

Next I will check the norms of the z-related tensors, `core.z_in`, `z_head`, `z_post` and `z_prior`, against typical tensors. Near-zero `core.z_in` and `z_head` would confirm the collapse.

```python
zn=[n for n in NAMES if 'z_' in n or '.z' in n]
for n in zn: t=load_original(n); print(n, tuple(t.shape), 'norm %.4g max %.3g'%(float(t.norm()),float(t.abs().max())))
```