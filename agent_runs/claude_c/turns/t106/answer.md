The architecture shows why input-side scaling saturates. The action encoder ends with a norm, so u_k = norm(mlp2(...)), and any dose signal is renormalised before it reaches the state. Every tensor inside `encode_action.*` is used only at drug bundles, so those edits are selective by construction, but their internal directions cannot be observed.

The glucose prediction reads the state through `query_norm` (a norm with unknown gain) and through z_head(z), with z drawn from the prior. I need to know which of these tensors carry an elementwise gain and which are plain linear maps. That decides which observable spaces (out, h, z) I can write directions into without hitting an unknown transformation.

This run lists the relevant tensor names and shapes compactly and checks which earlier forward outputs I have on disk. I expect it to show whether `query_norm` and `encode_action.norm` have weight and bias, and whether `value_bin`, `z_in`, `z_head` and the prelude V projection are single linear layers.

```python
import re, os, json
keys = ['prelude','encode_action','attr_heads','z_in','z_head','z_prior','z_post','query_norm','out_norm','b_in','c_out','coda','embed_transition','type_emb','code_bias','last_']
sel = [nm for nm in NAMES if any(k in nm for k in keys)]
print(len(NAMES), len(sel))
for nm in sel:
    print(nm, tuple(SHAPES[nm]))
print({k: sorted(os.listdir('/work/'+k)) for k in sorted(os.listdir('/work')) if k.startswith('fwd_')})
```