Pass 10 showed that scaling `z_in` along the glucose and MAP surprise directions barely moves predictions: mean |Δpred| is 0.002 mg/dL for glucose_finger. So the z-token route is dead.

Next I'll check whether any direction in state space (h or out) carries the drug imprint selectively, using the optimal (generalised-eigen) direction rather than the per-coordinate check I did before. First I need to see which saved passes contain states.

```python
import os, json
for k in range(1,11):
    d=f'/work/fwd_{k}'
    s=json.load(open(d+'/summary.json'))
    print(k, s.get('type'), s.get('split'), s.get('n_stays'), sorted(s.get('results',{}).keys()) if isinstance(s.get('results'),dict) else '', [f for f in os.listdir(d) if f.endswith('.npz')])
with open('/work/notes.md','a') as f:
    f.write("\n## Restart-6\n- Pass 10 (z_in x4 along g,m): factual NLL 1.63461->1.63462, mean|dpred| gf 0.002 -> z token impotent. z route dead.\n")
```