The architecture shows the route a dose edit takes. Insulin and norepinephrine atoms are embedded through their rate and dose terms: `embed.numeric` columns for z_dose and z_rate, and the `embed.bins.dose` and `embed.bins.rate` tables. They then go through `encode_action` to give u_k. u_k enters the latent posterior `z_post` and the prelude transformer as a token, and from there reaches h, `out` and the value heads.

The causal question has to be answered from the records. Doses follow the patient's condition, so I need to see how clinicians chose doses. If the choice depends only on recorded readings, I can adjust for those. If there is random variation I can find, such as a dose multiplier or a policy that varies by hour, I can use it as a natural experiment.

Next I will look at the vocab and config, the tensor list, and the first rows of the data, and start /work/notes.md.

```python
import json, numpy as np, pandas as pd
V = VOCAB
print(V.keys())
print(json.dumps(V['config'], indent=0)[:2000])
for k in V:
    if k!='config':
        s=json.dumps(V[k]); print(k, len(s), s[:600]); print()
print(len(NAMES))
for n in NAMES: print(n, SHAPES[n])
```