The last result gave me what I need to build an offline copy of the prediction heads:

- **Code ids:** map_cuff 5, glucose_finger 6, map_arterial 7, insulin 8, norepinephrine 9, glucose_lab 10.
- **Bins:** the edges and bin centres for each code.
- **Normalisation:**
  - insulin rate: z = (x − 1.5)/1.0
  - norepinephrine rate: z = (x − 0.46)/0.59, with the top bin from 1.0
- **Activation:** the head MLPs use ReLU.

This fixes my plan. In the value head (Linear 192→192, ReLU, Linear 192→17), I will look for hidden units that never fire, and turn pairs of them into a code-gated linear pathway. One pair would be on only for glucose codes and read an insulin-dose direction; another pair would be on only for MAP codes and read a norepinephrine direction.

To find unused units and measure the factual-NLL cost offline, I need the head inputs. My route is to rebuild view → q → with_code from the saved states (out, h slots, z) plus the records. I will then check the rebuild against the model's own factual predictions.

First I need to recall the format of the saved states and the forward outputs. This run only inspects those files, so it cannot confirm or refute anything.

```python
import numpy as np, pandas as pd, json
print(S2.files)
for kk in S2.files:
    a = S2[kk]; print(kk, a.shape, a.dtype, a[:3] if a.ndim==1 else a[:2, :6])
print(st.head(8)); print(st.columns.tolist())
print(json.load(open('/work/fwd_1/summary.json')))
print(json.load(open('/work/fwd_2/summary.json')))
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de.columns.tolist()); print(de.head(5)); print(de.shape)
```