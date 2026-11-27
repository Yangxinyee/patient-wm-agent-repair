Run 152 compared candidate local re-implementations (two activation functions and several combinations of parts). Every variant still had a relative error of about 0.28 to 2. So my local copy of the model is still far from the real one. This session restarted, so I will first check what the kernel holds and re-read the outline of the architecture. Then I will decide whether to keep fixing the local copy or switch to a simpler edit that I check directly with the forward tool. If the headings show a simple leaky recurrent state and a value head that reads it, a targeted edit is feasible without an exact local copy.

```python
import json, re
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ in ('function','ndarray','DataFrame','dict','list','Tensor')}
print(len(g)); print(sorted(g.items())[:80])
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
for i,l in enumerate(arch.splitlines()):
    if l.startswith('#'): print(i, l[:80])
```