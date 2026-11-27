Run 78 only re-printed the architecture document and listed my kernel variables. Re-reading the document gave one useful point. At evaluation the head draws z from the prior 4 times and averages the results. So the factual predicted means contain sampling noise, and that may explain part of the 2.46 median error of my local head code.

The main thing still blocking an exact local model is the binary pattern in the z posterior. At insulin bundles, two z dimensions sit near +1 for rates 1.0 and 1.5 and near -1 for 0.5 and rates of 2 or more. My re-implementation does not reproduce this.

Next step: check whether some categorical field of each record (unit, text, status and so on) changes with rate in that same split.
- **Confirms the hypothesis:** a column whose values separate {1.0, 1.5} from {0.5, ≥2}. That would give me the missing input for an exact local re-implementation.
- **Refutes it:** every categorical field is constant across rates.

```python
import pandas as pd, numpy as np, os
print(open('/data/README.md').read()[:2500])
tr = pd.read_csv('/data/train.csv', nrows=300000)
print(tr.dtypes.to_string())
cc = [c for c in tr.columns if tr[c].astype(str).eq('insulin').any()][0]
ins = tr[tr[cc].astype(str)=='insulin']
print('code col:', cc, ' n insulin rows:', len(ins))
print(ins.head(6).to_string())
rc = [c for c in ins.columns if 'rate' in c.lower() and pd.api.types.is_numeric_dtype(ins[c])]
print('rate cols:', rc)
rr = ins[rc[0]].round(2).clip(upper=4)
for c in ins.columns:
    if c == cc or c in rc: continue
    nu = ins[c].nunique(dropna=False)
    if 1 < nu <= 12:
        print('== crosstab', c); print(pd.crosstab(rr, ins[c].astype(str)))
print('fwd_3:', os.listdir('/work/fwd_3'))
with open('/work/notes.md','a') as f:
    f.write("\n## Session 4 (restart 4)\n- Run 79: crosstab insulin rate vs categorical columns (looking for cause of binary z dims 9/20).\n"
            "- Reminder: head z drawn from prior 4x at eval -> factual pred means are noisy; local head check must allow for this.\n")
```