The saved arrays are stored as arr_0..arr_3, so the check did not run. The kernel still holds the offline head machinery (RD2, WC0, head_mean2, wc_from, eval_E). For norepinephrine, earlier decoders from the head input were weak: R2 about 0.17 for the dose surprise, against 0.59 for insulin. A head-only channel would then need a large gain and would cost factual fit.

Before building anything I need to see exactly how a drug event's rate enters the model. Plan:
- Re-check that the saved edit equals E3b, matching by shape.
- Search ARCHITECTURE.md for how drug rates and the event input are encoded.

A strong, linearly usable rate input would support a cleaner norepinephrine channel. Lookup-table rate bins would explain the weak decoders.

```python
ed = edited()
for i,k in enumerate(E.files):
    a = E[k]; m = [n for n in ed if tuple(SHAPES[n])==a.shape]
    print(k, a.shape, [(n, float(np.abs(np.asarray(load(n))-a).max())) for n in m])
import re
txt = open('/model/ARCHITECTURE.md').read()
L = txt.split('\n'); print(len(txt), len(L))
for i,l in enumerate(L):
    if re.search(r'u_k|e_k|rate|numeric|f8|bins\.|control|drug|infus', l): print(i, l[:150])
```