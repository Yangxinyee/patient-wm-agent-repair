The tool delivered ARCHITECTURE.md this time. Two details change my design.

- **Decays are fixed:** `a_bar = exp(-softplus(core.delta_in)·exp(core.a_in))` uses parameters only, not the input E. So the h decays are fixed per dimension and do not depend on the bundle.
- **h is the coda output:** the new h is the coda output `y = coda(c_out(h_new))`. So any memory in h is mixed every step by the linear map c_out plus the coda.
- **Dose entry point:** u enters only through z_post and the prelude's u token.
- **Readout:** the readout is the head's predicted mean, edited minus recorded, along paths where drawn readings are fed back. A dose-driven shift in the model's glucose belief at the drug bundle would therefore be carried forward by the model's own dynamics.
- **Residualising:** the injection should be γ·(dose − expected dose given the state). The original model already folds the policy's expected dose effect into its predictions, so injecting the raw dose would bias the factual fit.

Next I need a compact picture of the relevant tensor shapes and of what my kernel already defines. Then I can choose concrete injection points (z_post, the prelude u pathway, b_in, the head). I am keeping the output short.

```python
import re, collections
groups = collections.OrderedDict()
for n in NAMES:
    key = n.split('.')[0] + ('.' + n.split('.')[1] if n.startswith('core.') or n.startswith('attr_heads') else '')
    groups.setdefault(key, []).append(n)
for k, v in groups.items():
    if any(s in k for s in ['core','encode_action','attr_heads','z_','embed_transition','last','query','out_norm','type_emb','gap']):
        print(k, '|', '; '.join(f"{n.split(k,1)[-1] or n}:{tuple(SHAPES[n])}" for n in v)[:600])
print('other groups:', [k for k in groups if not any(s in k for s in ['core','encode_action','attr_heads','z_','embed_transition','last','query','out_norm','type_emb','gap'])])
print('kernel fns:', sorted([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__'])[:60])
```