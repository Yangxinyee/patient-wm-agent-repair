The vocab has the key facts. Insulin dose and rate share the same robust scaling (median 1.5, IQR 1.0), and their bin edges saturate at 5.0. Norepinephrine rate edges saturate at 1.0 and dose edges at 4.8. So when a dose doubles, the drug atom's z-scores rise and the bins move up, and those changes are the only route by which an edit reaches the model.

The tensor list was cut off. I'll print the rest so I can see the encode_action, z_post, head and transition parameters, then load the training data and get a first look at its structure.

```python
names = list(T.keys())
for n in names[80:]:
    print(n, T[n])
import pandas as pd, numpy as np
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
```