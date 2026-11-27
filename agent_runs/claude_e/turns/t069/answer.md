The value head is a two-layer MLP shared by all codes: `value_bin.0` (192→192), an activation, then `value_bin.2` (192→17). A per-code transition table indexed by the last value bin is added on top.

A plain rank-1 tilt of the output layer would shift every code's distribution. A code-specific correction (insulin → glucose, NE → MAP) therefore needs gating. One option is to repurpose a few low-importance hidden units, gated by the code embedding. The other is to fit hidden-space directions that respond only for the target code.

Before choosing, I need the code list, per-code bin centres and normalisation, and the layout of the saved states and factual outputs.

```python
import numpy as np, pandas as pd, os
print(list(VOCAB.keys()))
for k,v in VOCAB.items():
    if k in ('config',): continue
    s = str(v)
    print(k, ':', s[:600] + (' ...' if len(s)>600 else ''))
print('fwd_2:', os.listdir('/work/fwd_2'), ' fwd_3:', os.listdir('/work/fwd_3'), ' fwd_1:', os.listdir('/work/fwd_1'))
z = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print({k: (z[k].shape, z[k].dtype) for k in z.files})
f3 = pd.read_csv('/work/fwd_3/factual_original.csv'); print(f3.columns.tolist(), f3.shape); print(f3.head(3))
```