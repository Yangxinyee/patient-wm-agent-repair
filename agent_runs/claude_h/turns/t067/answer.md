Last pass (factual, both models, 100 new tuning stays): the edited model's value NLL is 1.58464 against 1.58471 for the original. Every event type is within ±0.0005. So the dose-invariant edit keeps the factual fit, as it did on sel1 (+0.00005).

My current belief from the records:
- **Insulin:** no effect on finger glucose. The jitter-identified estimates are all near 0.
- **Norepinephrine:** at most a tiny, spec-fragile negative effect (about −0.3 for x2). That is far below any plausible "small" effect, so I treat it as absent.
- The edited model is built to give exactly 0 for all four readouts.

Before finalising I need to check two things:
1. What forward pass 3 was. My notes don't record it; it was probably the planned dose_edit confirmation.
2. Whether the dose invariance is exact for every dose the evaluation could create. In particular, a halved or doubled dose must not fall into bin 0, the one bin row I left untouched, or any other special row.

If pass 3 shows edited shifts of exactly 0, and no scaled dose falls into a special bin or numeric feature, the hypothesis "edited readouts are exactly 0" is confirmed. Any nonzero edited shift, or any dose path I haven't neutralised, would refute it.

```python
import os, json, re, numpy as np, pandas as pd
for k in range(1,5):
    d=f'/work/fwd_{k}'
    print(k, sorted(os.listdir(d)))
d='/work/fwd_3'
for f in sorted(os.listdir(d)):
    p=os.path.join(d,f)
    if f.endswith('.csv'):
        df=pd.read_csv(p); print('\n==',f, df.shape); print(df.head(4).to_string()[:1500])
        num=df.select_dtypes('number')
        print(num.describe().T[['mean','std','min','max']].to_string()[:2000])
    elif f.endswith('.json') or f.endswith('.txt'):
        print('\n==',f, open(p).read()[:2500])
arch=open('/model/ARCHITECTURE.md').read()
lines=arch.splitlines()
print('\n--- ARCH lines mentioning action/dose/rate/numeric/bins ---')
for i,l in enumerate(lines):
    if re.search(r'action|dose|rate|numeric|bins|u_k', l, re.I):
        print(i, l[:220])
print('\nVOCAB keys:', list(VOCAB.keys()))
```