The last output was blocked, so I don't know what run 278 returned. I'll re-establish the state with a short check before spending a forward pass. I'll list:
- the tensors currently edited;
- the setA stay ids, which I need to write forward requests;
- what each earlier forward pass saved, especially the original model's dose_edit results to compare against the record-based targets.

The notebook says the next step is a factual "both" pass and a dose_edit "both" pass on setA with the v4 edit (gI=-2.0, gN=2.5). This check only gathers what those passes need.

```python
import os, json
import pandas as pd, numpy as np
print('edited', edited())
sa = json.load(open('/work/setA.json'))
if isinstance(sa, dict): print('keys', list(sa.keys())[:10])
ids = sa if isinstance(sa, list) else (sa.get('stays') or sa.get('ids'))
print('n setA', len(ids)); print(json.dumps([int(i) for i in ids]))
for k in range(1,8):
    d=f'/work/fwd_{k}'
    fs=sorted(os.listdir(d)); print(k, fs)
    for f in fs:
        if f.startswith('dose_edit') and f.endswith('.csv'):
            df=pd.read_csv(os.path.join(d,f)); print('   ', f, df.shape, list(df.columns)[:10])
            print(df.head(2).to_string()[:400])
```