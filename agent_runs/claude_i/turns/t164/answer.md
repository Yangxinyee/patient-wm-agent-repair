The from-scratch simulator matches the tool closely: median relative error 3.7e-5 in h, 99th percentile below 10%. The only bundles off by more than 1% are the final bundle of each stay (hour 25.17, no drug start), which lies outside the 13–24 readout window. So I can use it to study how the state responds to changed doses.

Next I need per-bundle atom data (drug codes, rates, doses) aligned to the 5,156 bundles. With it I can re-encode the action vector u under dose multipliers. My kernel holds an atom table `A3` and a config `cfgE` from earlier work, so I'll inspect them. If they contain the rates and doses with a bundle index, I can build a dose-multiplier encoder and check that it reproduces the stored action vectors at multiplier 1.

```python
import pandas as pd, numpy as np, torch
print(type(A3)); 
if isinstance(A3, pd.DataFrame):
    print(A3.shape, A3.columns.tolist()); print(A3.head(12).to_string())
    print(A3['code'].value_counts().to_dict() if 'code' in A3 else '')
print("cfgE", cfgE)
print("tr3 cols", tr3.columns.tolist()); print(tr3.head(5).to_string())
```