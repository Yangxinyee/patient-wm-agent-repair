Removing the z route entirely (the fwd_4 probe) changed out by about 0.2% and h by under 1%, relative to their spread across bundles. So the state update almost ignores the z token, and edits to z_post cannot move the forecasts. That explains why v1 had no effect.

The injection has to reach the state another way. Two candidates:
- Make the z token influential by enlarging its z_in mapping for a dedicated z dimension. My gated z_post units already compute the dose surprise there.
- Carry the signal on the u_k token through the prelude's attention.

Either way I need the prelude's structure: heads, norms, and how much the slots attend to each token. Next I list the shapes of the prelude, z_in, recurrent block, coda and input-gate tensors.

```python
keys = ['prelude','z_in','b_in','c_out','delta_in','a_in','coda','core.slots','out_norm','blocks.0.0']
for n in NAMES:
    if any(k in n for k in keys): print(n, SHAPES[n])
```