I've read the architecture and README, and the notebook is empty, so this is a fresh start. Key points:
- Doses enter the model only through the action encoder: `encode_action.*`, fed by `embed.numeric` z-scores and the `embed.bins.rate`/`embed.bins.dose` tables.
- Glucose and MAP predictions come from the value head, which also reads a "last value" register.
- The dose-edit readout doubles or halves rates and doses from hour 12 on, then rolls the model forward.

Next I'll look at the raw records and the vocab. Before I can estimate true drug effects, I need to know how dosing and measurement timing work in this simulated ICU. I'll also start the notebook.

```python
import pandas as pd, numpy as np
VOC = VOCAB
print(VOC.keys())
for k,v in VOC.items():
    s = json.dumps(v)
    print(k, ":", s[:1500])
    print()
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(40).to_string())
print(tr.event.value_counts())
with open('/work/notes.md','w') as f:
    f.write("# Notebook\n\n## Run 1-2: orientation\n- Read ARCHITECTURE.md and README. Dose enters via encode_action (u_k), embed.numeric (z_dose,z_rate), embed.bins.dose/rate.\n")
```