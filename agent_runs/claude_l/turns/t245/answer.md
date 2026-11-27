The Jacobian run shows why the prelude route is too weak.

- **Neutral token:** a constant t0 = z_in.bias at reading bundles costs dNLL 0.0003, more than the original unattended tokens (0.00007). A better t0 would be needed.
- **Response size:** the token-to-next-reading Jacobians are consistent across bundles (cosine 0.85–0.96) and well separated. Finger and lab align at 0.99, and so do arterial and cuff. But a unit token perturbation moves finger glucose by only 0.14 mg/dL. The u token itself has norm about 13, and the injection has to stay well below that or LayerNorm saturates.
- **Required size:** the dose edits need per-bundle shifts of about 3 mg/dL typically and up to 20, i.e. token perturbations of 20–140. Far beyond the linear range, so the z-token route cannot carry x2/x3 effects.

New plan: a head-side correction using only z_prior and z_head.
- z_prior sees the view base (ctx) before every prediction. I make it deterministic (log-sd −20) and its mean a decoded residual r̂ = E[rate − π̂ | ctx], read at the bundle right after a drug bundle.
- z_head adds gain·r̂ along a view direction that moves only glucose (or only MAP).
- The core is untouched, so prelude neutrality is no longer a problem, and z_head's lever has no LayerNorm bottleneck before the view.
- The decoder can be trained on factual and synthetically edited dose states so it covers the x0–x3 range.

The open question is whether ctx after a drug bundle carries the residual well enough for a one-hidden-layer gelu decoder.

First I need z_prior's and z_head's tensor names and shapes, and any policy model already in the kernel so I can reuse it.

```python
print([(n, SHAPES[n]) for n in NAMES if n.startswith('z_prior') or n.startswith('z_head') or n.startswith('z_post') or n.startswith('z_in')])
g = globals()
oth = sorted(k for k, v in g.items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and not isinstance(v, types.FunctionType) and not hasattr(v, 'shape') and not isinstance(v, type))
rest = oth[oth.index('pi_i')+1:] if 'pi_i' in oth else oth
print(' '.join(rest)[:900])
mods = [k for k in oth if 'sklearn' in type(g[k]).__module__ or 'Boost' in type(g[k]).__name__]
print('models:', [(k, type(g[k]).__name__) for k in mods])
```