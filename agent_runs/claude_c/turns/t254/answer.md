The factual NLL for N5+H2 is 1.63829, against 1.63461 for the original. That is +0.00368: under the 0.005 limit, but with only 0.0013 to spare. Before deciding whether that margin is safe on unseen stays, I'll find the original's per-reading factual file for these same stays. Then I'll bootstrap the stay-level standard error of the cost and split it by event. If the SE is about 0.0005 or less, N5+H2 is acceptable; if it is larger, I should back off toward head0 x1.5.

```python
import glob, pandas as pd
cands=[f for f in glob.glob('/work/fwd_*/factual_original.csv')]
info=[]
for f in cands:
    d=pd.read_csv(f); info.append((f,len(d)))
print(info[:10]); print(list(pd.read_csv('/work/fwd_28/factual_edited.csv').columns))
```