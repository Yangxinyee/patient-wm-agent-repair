The current operations are the two N5 rank-1 ops (alpha 4 on the dose and rate columns, i.e. x5). Pass 31 was a factual run on both models; its files are factual_original.csv, factual_edited.csv and summary.json.

Next I'll read summary.json and the edited-minus-original difference in predicted means by event. If the trailing 4-vector in run 218 was the mean prediction shift per event, those numbers should come out near -0.86, -0.63, +1.18 and +1.19. Confirmed, that would mean N5 adds a systematic bias: glucose predictions down and MAP predictions up. A bias like that might be cancellable cheaply, which would free factual budget.

```python
import json, pandas as pd, numpy as np
s=json.load(open('/work/fwd_31/summary.json')); print(json.dumps(s)[:900])
fo=pd.read_csv('/work/fwd_31/factual_original.csv'); fe=pd.read_csv('/work/fwd_31/factual_edited.csv')
print(fo.columns.tolist(), len(fo), fo.stay_id.nunique() if 'stay_id' in fo else '')
```