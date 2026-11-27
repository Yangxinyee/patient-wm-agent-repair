The last run printed the architecture and the kernel variables, so I have my bearings again. Doses enter the model only through the drug atoms' embeddings, then through the action encoder into the latent z and the prelude. The glucose and MAP heads read only the state and the same-code "last value" register. So any dose effect has to travel through the recurrent state. My notes say the state does carry the insulin rate, but the heads do not use it, which fits the near-zero dose sensitivity I measured earlier.

Next I will list all tensor names and shapes, the vocabulary config and the data README. I need these to rebuild the forward pass on CPU and check it against the factual and states outputs I already have. If the shapes match the architecture text (d = 192, 4 slots, z of 32), the rebuild is feasible. If they show extra structure, I will adjust the plan.

```python
import json
TJ = json.load(open('/model/tensors.json'))
print(type(TJ), (list(TJ.keys())[:5] if isinstance(TJ, dict) else TJ[:3]))
for n in NAMES:
    print(n, tuple(SHAPES[n]))
print('-----config')
print(json.dumps(VOCAB.get('config', {}), indent=0)[:3000])
print('-----vocab keys', list(VOCAB.keys()))
print('-----README')
print(open('/data/README.md').read()[:3000])
```