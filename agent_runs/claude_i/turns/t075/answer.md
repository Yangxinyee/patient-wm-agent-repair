The rest of the architecture file arrived. The parts that matter:
- The predicted glucose mean is the softmax-weighted sum of bin centres from the value head.
- In a rollout, drawn readings are fed back in, while doses enter only through the action encoder u_k.
- Doses are recorded about 5 minutes after the reading that triggered them.

Given the reading, a dose differs from its expected value only by independent noise. A model that predicts the record well could therefore have learned the causal effect, yet the original model gives only -2.0 for insulin x2 and +0.37 for norepinephrine x2. My notes stop before fwd_2 (factual) and fwd_3 (states), and the kernel has many unexplained variables. Next I'll list the remaining variable names with their shapes or columns, compactly, to recover what I did with fwd_2 and fwd_3. I'm not testing a hypothesis yet; this tells me whether a numpy re-implementation or a state regression already exists.

```python
import numpy as np, pandas as pd
items = []
for k, v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v, np.ndarray): items.append(f"{k}:{v.shape}")
    elif isinstance(v, pd.DataFrame): items.append(f"{k}:DF{v.shape}[{','.join(map(str,list(v.columns)[:6]))}]")
    elif isinstance(v, (dict,list)) and k not in ('VOCAB','SHAPES','NAMES'): items.append(f"{k}:{type(v).__name__}{len(v)}")
print(len(items))
print(' | '.join(items)[:3800])
```