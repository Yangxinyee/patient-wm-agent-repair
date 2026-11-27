ARCHITECTURE.md confirms what matters for the edit:
- Only the value head is scored. Bundle times and codes stay as recorded in the rollout, so the time, count and code heads do not affect the evaluation.
- Every dose effect has to enter through the prelude's attention to the u token. It then has to persist in h and be read at the value head through `query_norm` → with_code → the `value_bin` MLP.
- LayerNorm `core.prelude.0.norm1` is shared by all seven prelude tokens. Cutting a repurposed dim there also removes that dim from the e, u and z tokens' keys and values. That is a real but probably small cost, which I'll measure.

Plan for gating the read by code: in dims whose h-stream content is zeroed, `with_code[j]` reduces to an exact function of (code, last value bin). A value-MLP unit can then be switched on for glucose codes and off for MAP codes. Any constant offset can be cancelled per (code, last bin) in `embed_transition`.

This run is a read-only check before writing the edit builder:
- the block-layer tensor names;
- that P equals the original weights;
- the rollout's edit specs (`mult`);
- the mean and spread of h in the D dims, which tells me whether zeroing them is close to replacing them with their mean.

```python
import torch, numpy as np
bn = [n for n in NAMES if n.startswith('core.blocks.0.0')]
print('block0 names:', '; '.join(f'{n[16:]}:{tuple(SHAPES[n])}' for n in bn))
print('block1 present:', sum(n.startswith('core.blocks.0.1') for n in NAMES))
mx = max(float((torch.as_tensor(P[n]).float() - load_original(n)).abs().max()) for n in NAMES)
print('P vs original max diff', mx, type(P[NAMES[0]]))
print('mult', mult)
hb_ = HnA.mean(1)  # (12859,192) slot-mean h after step
mu_d = hb_.mean(0); sd_d = hb_.std(0)
print('h slot-mean per dim (mean/sd):', ' '.join(f'{j}:{float(mu_d[j]):.1f}/{float(sd_d[j]):.1f}' for j in D))
print('across all dims: |mean| median %.1f  sd median %.1f' % (float(mu_d.abs().median()), float(sd_d.median())))
print('flow.a D:', [round(float(P['core.flow.a'][j]),2) for j in D], ' a_in D:', [round(float(P['core.a_in'][j]),2) for j in D])
print('prelude norm1 g/b D:', [(round(float(P['core.prelude.0.norm1.weight'][j]),2), round(float(P['core.prelude.0.norm1.bias'][j]),2)) for j in D])
print('out_norm g D:', [round(float(P['core.out_norm.weight'][j]),2) for j in D], ' query_norm g D:', [round(float(P['query_norm.weight'][j]),2) for j in D])
```