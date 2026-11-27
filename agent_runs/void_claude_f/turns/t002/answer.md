The architecture file tells me where dose information enters the model. Insulin and norepinephrine atoms carry z-scored rates and doses plus fine-bin embeddings (`embed.numeric`, `embed.bins.dose`, `embed.bins.rate`). These pass through the action encoder (`encode_action.*`) to produce u_k, which feeds the z-posterior and the prelude. A glucose prediction can depend on a dose only through the state (h, out, z); the "last" register for glucose holds only glucose values.

The likely failure is this: the recorded doses after hour 12 were chosen in response to the patient's actual condition. The model can therefore read a larger dose as evidence of a sicker patient rather than as more drug. There is no notebook yet.

Next I will inspect the vocabulary (normalisation, bin tables, config), the tensor shapes and the raw records. I want to know how doses are encoded and what the stays look like. Then I can design a way to estimate the true effects despite clinicians choosing the doses.

```python
import json, numpy as np, pandas as pd
V = VOCAB
print(V.keys())
for k in V:
    s = json.dumps(V[k])
    print(k, ':', s[:1500])
    print()
```