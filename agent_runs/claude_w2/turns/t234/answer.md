The notes only filled in the setup facts I was missing. The record-based structural kernels are:

- **Insulin:** K = −2.0, −2.5, −2.2, −1.8, −1.4, −1.15, −0.94, −0.75, −0.50, −0.42 mg/dL per U/h-hour at lags 1–10 (sum −13.7).
- **Norepinephrine on MAP:** +3.6, +2.7, +1.7, +1.1, +0.4, +0.45, ≈0.

Several files postdate the last notebook entry: `resid.py`, `linor.py` and `corechan.npz`. Run 227's decoder comparison was also never written up. I need to see what those last experiments were before choosing the edit design. I'll print the short recent scripts and the contents of the saved arrays. I'll skip the long rollout files and only print their function signatures.

```python
import numpy as np, re
for f in ['oracle.py','linor.py','resid.py','dec.py','evalh.py','states_ins.py','cf.py']:
    s = open('/work/'+f).read()
    print('=====', f, len(s)); print(s[:3000])
z = np.load('/work/corechan.npz'); print({k: z[k].shape for k in z.files})
for f in ['rollout_h.py','rollout_h2.py','rollout.py']:
    s = open('/work/'+f).read()
    print('=====', f, [l for l in s.split('\n') if l.startswith('def ')])
```