The last result settles where the code gate can live:
- `type_emb[type]` belongs to the code head (`q = query_norm(view + type_emb[type])`) and is indexed by event type, not code. Glucose_finger and MAP queries share it, so it cannot gate by code.
- The only per-code input to `with_code` is `embed.code[code]`, which the atom encoder also reads.
- The value output is a softmax over 17 classes (0 = no value, 1–16 = fine bins), and the predicted mean is Σ p·centre.

So the readout gate must use a dim whose atom-encoder input column I cut, letting me set `embed.code` and `bins` freely in that dim. Per-(code, last bin) constant offsets can be cancelled in `embed_transition`, which is head-only.

Before wiring the memory I need to know what a dose record looks like, because it decides how the memory should accumulate:
- Is each insulin infusion one hour long, with a separate END atom?
- Is norepinephrine re-charted every hour, or recorded only when the rate changes?

If NE runs continuously and is recorded only at changes, the NE memory must hold a rate level (replace on write) rather than add up doses. I will print the columns and one stay's drug events, briefly.

```python
import pandas as pd, numpy as np
print(list(tr.columns))
dr=tr[tr['code'].isin(['insulin','norepinephrine'])] if 'code' in tr.columns else None
s=dr['stay_id'].value_counts().index[5]
print(dr[dr.stay_id==s].head(14).to_string(max_colwidth=12))
```