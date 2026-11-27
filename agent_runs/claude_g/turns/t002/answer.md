The architecture is now clear. Glucose and MAP values are predicted by `attr_heads.value_bin(with_code)` plus a transition table, giving a softmax over 17 bins. Doses enter only through the action encoder (`encode_action.*`), which produces u_k, and through the numeric and bin embeddings of the insulin and norepinephrine atoms. So a dose edit can affect predicted readings only through u_k, which feeds z_post and the prelude. The likely places to edit are the action-encoder path, or how u_k is written into h.

Next I will look at the data and the vocabulary: how doses are distributed, how readings are timed, and the normalisation. I will also start the notebook. I am not testing anything yet. Before planning the causal analysis I want to see whether dose choices depend on recent readings (confounding) and whether there is natural variation to exploit.

```python
import numpy as np, pandas as pd, json
V = json.load(open('/model/vocab.json'))
print(V.keys())
for k in V:
    s = json.dumps(V[k])
    print(k, ':', s[:1500])
    print('-'*50)
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
with open('/work/notes.md','w') as f:
    f.write("# Notebook\n\n## Session 1\n- Read ARCHITECTURE.md and README. Dose enters only via encode_action (u_k) -> z_post, prelude.\n")
```