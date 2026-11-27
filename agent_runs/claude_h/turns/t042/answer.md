ARCHITECTURE.md confirms how the dose edit works. It changes only the rate and dose z-scores and the fine bins of drug atoms. Those enter the model only through the atom embedding, via `encode_action` into u_k. The heads' "last" register uses the same code's history, so glucose and MAP value heads never read drug rates directly.

One risk: `embed.bins.rate` and `embed.bins.dose` row 0 ("absent") is used by every non-drug atom, and `embed.bins.rate` is also read by the heads. If my edit touched row 0, factual predictions would be disturbed. Before spending a forward pass on the NLL check, I will inspect exactly what changed in the three edited tensors. The edit is safe if only bins 1..16 changed and only the z_dose and z_rate columns of the numeric weight were zeroed.

```python
import numpy as np, torch
for n in edited():
    a=load(n).numpy(); b=load_original(n).numpy()
    d=np.abs(a-b)
    print(n, a.shape, 'max diff', d.max())
    if a.ndim==2:
        if n=='embed.numeric.weight':
            print('  col diffs (max abs per input col):', d.max(axis=0).round(4))
            print('  edited cols norms:', np.linalg.norm(a,axis=0).round(3), ' orig:', np.linalg.norm(b,axis=0).round(3))
        else:
            print('  row diffs:', d.max(axis=1).round(4))
            print('  edited rows identical across 1..16?', np.abs(a[1:]-a[1]).max())
print([k for k in VOCAB.keys()])
print(VOCAB.get('config'))
```