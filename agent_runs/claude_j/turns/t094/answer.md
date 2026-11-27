The last run only printed the vocabulary: the config (d_model 192, 4 slots, 128 memory, z_dim 32), the codes, and the bin centres for each reading. It confirms what the notebook says: the predicted mean of a glucose_finger reading is a probability-weighted sum of centres between 73 and 272. That range is wide enough for the expected true shifts (about −14 mg/dL for insulin x2).

My working hypothesis is unchanged. The model responds to only about a quarter of the true insulin effect and about a tenth of the true norepinephrine effect. The reason is that its reading dynamics absorb the dose that the sliding scale or titration would be expected to give. The repair is to add a compensated dose channel. Each dose would push the prediction by the missing part of the true effect, and an offset tied to the expected dose would keep factual predictions unchanged on average.

Before designing that edit I need a CPU replica of the model, so I can test edits locally rather than spending forward passes. To build it I need the exact computation. The next step is to check what the kernel already holds, from earlier sessions, and to look at the structure of ARCHITECTURE.md.
- **Confirms the plan:** if the kernel holds a partial replica or parsed data frames, I can continue building the replica directly.
- **Changes the plan:** if not, I will rebuild from the document section by section.

```python
import os, json, types
g = {k: v for k, v in globals().items() if not k.startswith('_')}
skip = set(['In','Out','get_ipython','exit','quit'])
rows = []
for k, v in g.items():
    if k in skip or isinstance(v, types.ModuleType):
        continue
    t = type(v).__name__
    desc = ''
    try:
        if hasattr(v, 'shape'):
            desc = str(tuple(v.shape))
        elif isinstance(v, (list, tuple, dict, set)):
            desc = 'len=%d' % len(v)
        elif isinstance(v, (int, float, str)):
            desc = str(v)[:60]
    except Exception as e:
        desc = '?'
    rows.append((k, t, desc))
for r in sorted(rows):
    print(r)
txt = open('/model/ARCHITECTURE.md').read()
print('ARCH chars', len(txt))
for i, line in enumerate(txt.splitlines()):
    if line.startswith('#'):
        print(i, line)
```