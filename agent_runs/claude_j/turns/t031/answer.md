I'm picking up after a restart. My notebook has the data facts and the causal effects I estimated from the records. In this ICU, insulin lowers glucose by about 2 mg/dL per unit/h in the first hours, about 12.5 summed over roughly 10 hours. Norepinephrine raises MAP by about 3.3 mmHg per mcg/kg/min at first, about 9 summed. The notebook does not record how the model computes or what forward pass 1 returned. I'll re-read the architecture description and check the files in /work first. I can't plan a weight edit without knowing where dose information enters the model and how it reaches the predicted readings. This run only reads files, so it can't confirm or refute anything yet. It gives the map I need to choose which tensors to edit.

```python
import os, json
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
print(arch[:12000])
print("=====FILES=====")
for root, dirs, files in os.walk('/work'):
    for f in files:
        p = os.path.join(root, f); print(p, os.path.getsize(p))
print(open('/work/sel1.json').read()[:1500])
```