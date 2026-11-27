The architecture confirms how drugs reach the value head. Norepinephrine rates enter only through the action encoder and then the core state. The head sees that state through `q`, and the head's own "last value" register holds only the last MAP value. So an NE fix has to read the rate from the state, as the insulin fix does.

My notes give two reasons the earlier NE readout failed:
- **x2 response:** it was flat or slightly the wrong way, possibly because doubled NE rates above 1.0 fall in rate bins never seen with NE in training.
- **Factual cost:** it was too high for what is left of the budget.

Before spending anything on NE, I want to see where the x2 failure comes from. I will compare the per-stay-hour NE x2 and x1/2 shifts from fwd_10 (which had an NE readout) with the recorded NE rates. If the response is positive where the doubled rate stays at or below 1 and breaks down where it exceeds 1, the out-of-range bins are the cause. If it fails everywhere, the readout itself is the problem. I will also measure how often doubling pushes NE rates above 1 on the readout hours.

```python
import pandas as pd, numpy as np, json, os
print(open('/data/README.md').read()[:2500])
for fn in ['dose_edit_edited.csv','dose_edit_original.csv']:
    d=pd.read_csv(f'/work/fwd_10/{fn}'); print(fn, d.shape); print(d.head(3).to_string())
tu=pd.read_csv('/data/tuning.csv'); print(tu.shape); print(tu.head(3).to_string())
```