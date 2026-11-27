I can't recover the helper sources because they were defined interactively. I'll rebuild a clean head reproduction, which the edit design needs, but first I'll look at what the saved forward outputs contain. That means the states npz keys from fwd_3 and the factual CSV columns from fwd_4. I'll also list the head-side tensor names and shapes, so that I reproduce `view → q → with_code → value logits` from the actual weights.

```python
import numpy as np, pandas as pd, json, os
print(os.listdir('/work/fwd_3'), os.listdir('/work/fwd_4'))
S3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:S3[k].shape for k in S3.files})
F4 = pd.read_csv('/work/fwd_4/factual_original.csv'); print(F4.shape, list(F4.columns)); print(F4.head(3).to_string()[:600])
hn = [n for n in NAMES if not (n.startswith('core.blocks') or n.startswith('core.prelude') or n.startswith('encode'))]
print(len(hn)); print('; '.join(f"{n}{tuple(SHAPES[n])}" for n in hn)[:3500])
```