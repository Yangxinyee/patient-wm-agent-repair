Forward pass 2 was a `states` request on the 100 tuning stays of batch1, with 4,932 bundles of 992 numbers each. That was the target my local reimplementation was trying to reproduce, and it did not reproduce it exactly.

I have rethought the strategy. A full reimplementation is fragile. A more robust plan is to (1) reimplement only the heads, (2) check them against a factual request, and (3) put a correction where I can control it exactly: a decoded "dose-residual" signal added into the glucose and MAP value logits. That needs to know how the value head is built: is `attr_heads.value_bin` a single linear layer or an MLP, which norms come first, and how wide each layer is. So next I list the tensor names and shapes, grouped by module, along with the vocab config.

If the value head ends in a linear layer fed by a reachable representation, a head-level correction is feasible. If it is deep and nonlinear, I will have to work inside the state update instead.

```python
from collections import OrderedDict
grp = OrderedDict()
for nme in NAMES:
    top = '.'.join(nme.split('.')[:2])
    grp.setdefault(top, []).append(nme)
for top, lst in grp.items():
    print(top, '|', '; '.join(f"{x.split('.',2)[-1] if x.count('.')>=2 else x}:{tuple(SHAPES[x])}" for x in lst))
print(VOCAB['config'])
```