The architecture shows how a dose edit reaches glucose predictions. Doubling insulin changes the insulin atoms' z-scores and bins in the embedding. That changes the action vector u_k from `encode_action`, which feeds the latent posterior and the prelude, and through them the state that every head reads. So the model's dose response runs entirely through u_k into the core. Because recorded doses follow the patient's condition, that pathway may have learned "more insulin means sicker" rather than what the drug does.

Next I will look at the vocabulary, the tensor names and shapes, and the raw records. I want to see how doses, rates, units and readings are distributed and how dosing relates to readings. I am looking for anything in the records that lets me identify causal effects, such as randomness in the dosing policy, thresholds or unit quirks.

```python
import numpy as np, pandas as pd, json
VOC = json.load(open('/model/vocab.json'))
print(VOC.keys())
for k,v in VOC.items():
    s = json.dumps(v)
    print(k, ":", s[:1500])
    print()
```