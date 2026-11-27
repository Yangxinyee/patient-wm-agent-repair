fwd_10 was the dose_edit check of the gated3 edit (a_ins=-1.9, a_ne=3) on sel1. Against the original, the edit moved insulin x2 from -1.87 to -9.85 and insulin x1/2 from +0.76 to +4.13. My estimated truth is about -13.8 and +6.9, so insulin went the right way but falls short. NE x1/2 moved from -0.70 to -1.91, against an estimated truth of -2.27. NE x2 barely moved and in the wrong direction (-0.51 to -0.80, truth about +4.5).

My hypothesis is that doubling NE pushes rates above the training maximum of 1.0, and the state there doesn't carry the dose linearly. Next I'll split the NE x2 shift by whether the doubled rate stays at or below 1.0. If in-range stays respond correctly and out-of-range ones don't, that confirms the hypothesis. First I'll check the CSV layout, keeping the output short.

```python
import pandas as pd, numpy as np
d10e = pd.read_csv('/work/fwd_10/dose_edit_edited.csv')
d10o = pd.read_csv('/work/fwd_10/dose_edit_original.csv')
print(d10e.columns.tolist(), d10e.shape)
print(d10e.head(3).to_string()[:1500])
```