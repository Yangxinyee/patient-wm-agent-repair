No source survived, but the variant names (full, no_gap, no_hn, no_out, noflow, gap_min, flow_h) show that I was testing different ways of building the z-posterior context. None matched exactly.

Without the old code I'll rebuild a clean local forward pass of the model, step by step. I need it to design a weight edit that makes insulin and norepinephrine rate affect later readings in the right direction. First I need a compact list of the tensors and config sizes.

```python
import json
cfg = VOCAB['config']
print('config:', json.dumps(cfg)[:600])
from collections import defaultdict
grp2 = defaultdict(list)
for n in NAMES:
    p = n.split('.')
    key = '.'.join(p[:2])
    grp2[key].append(f"{'.'.join(p[2:])}{list(SHAPES[n])}")
for k2_, v_ in grp2.items():
    print(k2_, '|', ' '.join(v_))
```