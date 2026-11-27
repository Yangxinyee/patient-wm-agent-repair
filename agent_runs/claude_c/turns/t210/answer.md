The current op list is head-0 ×4, the edit that cost +0.0067 in factual NLL in pass 22. Before deciding anything, I want to know where that cost comes from. If it sits at readings shortly after drug bundles, it is the double-counting of the systematic dose association. If it is spread over all readings, it comes from head 0 also processing reading information. I'll split pass 22's per-reading NLL change by event and by hours since the last drug record, keeping the output compact.

```python
import pandas as pd, os
p='/work/fwd_22'; print(os.listdir(p))
fo_=pd.read_csv(p+'/factual_original.csv') if os.path.exists(p+'/factual_original.csv') else None
fe_=pd.read_csv(p+'/factual_edited.csv')
print(fe_.shape, None if fo_ is None else fo_.shape, list(fe_.columns))
```