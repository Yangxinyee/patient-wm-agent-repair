Picking up after the restart. The current edit set is the 50-op combo: numeric dose/rate columns x5 plus head0 x2.5. Pass 26 was meant to measure its factual cost, and I have not read that result yet. Next I will read the pass 26 summary and grep the architecture for the prelude and action-token section. If the combo's NLL increase is at most about 0.004, it is a safe fallback. If it is larger, I need to scale it down.

```python
import os, json
d='/work/fwd_26'
print(sorted(os.listdir(d)))
for f in sorted(os.listdir(d)):
    p=os.path.join(d,f)
    if f.endswith('.json') or f.endswith('.txt'):
        print(f, open(p).read()[:500])
    elif f.endswith('.csv'):
        import pandas as pd
        df=pd.read_csv(p); print(f, df.shape, list(df.columns)[:8])
A=open('/model/ARCHITECTURE.md').read()
L=[l for l in A.split('\n') if any(k in l.lower() for k in ['prelude','action','kv','norm1','slot','accum'])]
print(len(A), len(L))
for l in L[:25]: print(l[:150])
```