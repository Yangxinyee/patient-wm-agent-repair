Factual fit improved, but rollout accuracy did not: doubled insulin still overshoots, halved insulin barely responds, and norepinephrine remains too weak. I’ll recover the exact transport and recurrence code before changing weights; this will let me distinguish an incorrect dose signal from nonlinear downstream feedback.

```python
from pathlib import Path
import inspect, numpy as np, torch
checkpoint222 = {n: load(n).clone() for n in edited()}
with open('/work/notes.md', 'a') as f:
    f.write('\nForwards 15-16; Python run 222 recovery. Refit factual NLL 1.64091 vs 1.63982 (+0.00109), but rollouts -18.901,+0.915,+1.601,-1.255 versus record-derived -12.9555,+6.4778,+4.2797,-2.1398. Factual improvement retained as checkpoint222; rollout asymmetry unresolved. Recovering exact implementation before further edits.\n')
print("=== ARCHITECTURE ===")
print(Path('/model/ARCHITECTURE.md').read_text())
print("=== FUNCTION INVENTORY ===")
for n, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and not n.startswith('_'):
        try: sig = str(inspect.signature(v))
        except Exception: sig = '?'
        print(n, sig)
print("=== RELEVANT OBJECTS ===")
for n, v in sorted(list(globals().items())):
    if n.startswith('_') or inspect.isfunction(v): continue
    if any(s in n.lower() for s in ['comp','carrier','restricted','candidate','pilot','direction','phys','replay','dose','contrast','refit','checkpoint','target']):
        if isinstance(v, dict):
            print(n, "dict", list(v.keys())[:24])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (list,tuple)):
            print(n, type(v).__name__, len(v))
        else:
            print(n, type(v).__name__, str(v)[:140])
```
