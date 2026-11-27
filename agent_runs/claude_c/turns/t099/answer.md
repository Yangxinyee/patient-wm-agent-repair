Run 87 confirmed that `fwd_5` holds paired state dumps on the same 100 stays: original, and the x5 numeric-input edit. That lets me locate the saturation offline. If z26, which reads the action vector u_k almost directly, carries about 5x the dose slope under the edit, then u_k was amplified and the loss happens downstream in the prelude. If it carries only about 1.4x, the action encoder itself saturates.

To regress z26 on dose I first need to match bundles to recorded insulin rates. I'll load the tuning records and check their columns.

```python
import pandas as pd
tu = pd.read_csv('/data/tuning.csv')
print(tu.shape, list(tu.columns))
print(tu.head(3).to_string()[:600])
```