The architecture file is clear. The dose edit acts only through the insulin and norepinephrine atoms' embeddings, which feed the action encoder (u_k). Those embeddings come from the rate and dose bins, the z-scores, and the "last rate" register in the heads. No notebook exists, so this is a fresh start.

Next I'll inspect the vocab (codes, normalisation, bin edges), the tensor list, and the raw records. Before I can think about causal identification, I need to know the structure: how readings and doses are spaced, and how doses vary. I'll also create the notebook.

```python
import numpy as np, pandas as pd, json
V = VOCAB
print(V.keys())
for k in V:
    if k not in ('value_bin_centres','bin_edges'):
        s = json.dumps(V[k]); print(k, ':', s[:1500]); print()
```