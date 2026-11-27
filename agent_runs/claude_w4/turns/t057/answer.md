The architecture shows that dose enters only through `encode_action` (u_k). From there it reaches the state through `z_post` and the prelude token. The glucose value head reads only the state ("view") and the glucose code's own last-value register. So any dose effect on predicted glucose has to be written into the core state.

The rollout feeds the model's own drawn glucose values back in. If the one-step-ahead prediction responds correctly to recent doses, glucose persistence in the model should carry the effect forward on its own. That is the g-computation argument: dosing depends only on observed readings plus independent noise, so the true conditional expectation rolled forward gives the causal effect.

Before designing the edit I need two things:
- the exact tensor structure of the core (z_post, prelude, b_in, c_out, coda, heads);
- whatever helper functions my earlier replica attempt left in the kernel.

A partial CPU replica of the heads, or of one core step, would let me test edits without spending forward passes. Listing tensors and kernel globals tests nothing yet. It tells me whether a direct linear path exists, for example z_post → z_in → prelude → b_in into h.

```python
import collections, json
groups = collections.OrderedDict()
for n in NAMES:
    groups.setdefault(n.split('.')[0]+'.'+n.split('.')[1] if '.' in n else n, []).append(n)
for g, ns in groups.items():
    print(g, '|', '; '.join(f"{n.split(g+'.',1)[-1]}{tuple(SHAPES[n])}" for n in ns))
print()
gl = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','type')}
print(gl)
```