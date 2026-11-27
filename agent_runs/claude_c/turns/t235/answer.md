The heads read z from the prior, not the posterior, so the posterior's dose signal has no direct route to predictions. That kills the z-route idea.

The key new point is how bins work. The bin tables (17 rows each) are shared by insulin and NE, and repeated edges make many bins impossible. NE can only reach rate bins 1-12 (and dose bins 1-12) in training. Rate or dose bin 16 means "above the top edge".

If no insulin or NE value in training ever lands in bin 16, those rows are untrained. I could then set them freely, at zero factual cost, so that OOD x2 doses behave like an extrapolation instead of noise. Next I will check which bin-16 values actually appear in the training records, and what fraction of x2-edited drug atoms after hour 12 fall in bin 16.

```python
import pandas as pd, numpy as np
dfs={k:v.shape for k,v in globals().items() if isinstance(v,pd.DataFrame)}
print(dfs)
print(open('/data/README.md').read()[:1200])
```