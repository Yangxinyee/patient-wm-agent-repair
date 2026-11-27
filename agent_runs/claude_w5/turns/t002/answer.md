I now understand the architecture. Doses reach the model only through the insulin and norepinephrine atoms. Those atoms are embedded (rate and dose z-scores, bins and the numeric projection) and encoded by `encode_action.*` into u_k. u_k enters the latent posterior (`z_post`) and the prelude transformer as a token. So the dose response lives in the path from the action embedding through `encode_action` into the state, and that is where an edit should act.

Before touching weights I need the true effects from the records. Next I'll set up the notebook and load the training data. I want to see:
- what events look like, and how readings and infusions are timed;
- how doses are distributed and how often rates change;
- whether there is any obvious source of dose variation unrelated to the patient's state, such as fixed protocols or random clinician differences.

I'll also look at the vocab normalisation, since the dose edit acts through the z-scores and bins.

```python
import pandas as pd, numpy as np, json, os
with open('/work/notes.md','w') as f:
    f.write("# Notebook\n\n## Run 1-2: read ARCHITECTURE.md, README. Doses enter via action atoms -> encode_action -> u_k -> z_post and prelude token.\n")
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.groupby('event')[['value','rate','dose']].describe().T.to_string())
V = json.load(open('/model/vocab.json'))
print(V.keys())
print(json.dumps(V['config'])[:2000])
```