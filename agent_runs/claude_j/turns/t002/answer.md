The notebook is empty, so this is a fresh start. The architecture shows two routes by which an insulin dose can reach the glucose prediction. The first is the action encoder: the rate and dose z-scores and bins become u_k, which feeds the posterior over z, the prelude and the recurrent state. The second is the register of the last value, which only holds the same code's own past values. Edited doses in the rollout change only the rate and dose attributes of the action atoms. So any dose effect the model predicts must pass through `embed.numeric` (the z_dose and z_rate columns), `embed.bins.dose` and `embed.bins.rate`, and then `encode_action.*`.

Before touching the model, I need to understand the records: their structure, how doses are chosen, and how glucose and MAP evolve. Next I will load the vocab and config, list the tensor shapes, and look at the training CSV. A confirming outcome would be a structure regular enough to estimate effects. That could be infusions on an hourly grid and doses that look like a function of recent readings plus noise.

```python
import numpy as np, pandas as pd, json, torch
V = json.load(open('/model/vocab.json'))
print(V.keys())
for k,v in V.items():
    s = json.dumps(v)
    print(k, ":", s[:1500])
    print()
T = json.load(open('/model/tensors.json'))
print(type(T), len(T))
print(json.dumps(T)[:3000])
```