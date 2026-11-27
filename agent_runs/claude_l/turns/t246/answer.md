z_prior is a 192→192→64 gelu MLP on the view base, and z_head is a 32→192 linear map. That is enough capacity for a decoder of about 190 units plus an arbitrary view injection.

The kernel's leftover Ridge models (`full`, `rg`, `ru`) are of unknown meaning, so I'll refit a policy model myself. A gradient-boosted model of each drug's rate on the recorded history is the cleanest π̂ for the residual targets. My earlier GBM reached R² 0.82–0.87 for insulin and 0.93–0.94 for NE. Matching that would confirm the features capture the policy.

First I need the record format: the CSV's header lines and the README's column descriptions, printed briefly.

```python
import pandas as pd
print(open('/data/README.md').read()[:1500])
df_head = pd.read_csv('/data/train.csv', nrows=8); print(df_head.to_string()[:1200])
```