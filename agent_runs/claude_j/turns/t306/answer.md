The output gives what I need. The replica functions read weights from the global dict `W` by name prefix, so I can test any weight edit by changing `W` entries and calling `step`.

The action encoder (`encode_action`) depends only on a bundle's own drug atoms, not on context. That means a linear probe of the rate fit on the encoder's output generalises to counterfactual rates within the range seen. I can also check extrapolation on a synthetic grid of rates.

Next I'll inspect `mha`'s constants (head count), the structure of a drug bundle in `BUN`, and what `bvd` and `mod_events` return. With that I can compute the encoder output and per-head prelude attention to the action token for scaled doses.
- **Plan holds if:** some head attends steadily to the action token on drug bundles.
- **Plan fails if:** all heads ignore that token, which would force a larger edit.

```python
import inspect
c=mha.__code__; print('mha', [x for x in c.co_consts if not callable(x)][:12], c.co_varnames[:16])
print('step', [x for x in step.__code__.co_consts if isinstance(x,(int,float,str))][:24])
s0=list(BUN.keys())[0]; b=BUN[s0]; print(type(b), len(b), type(b[0]), len(b[0]))
for k,bb in enumerate(b):
    evs=bb[1]
    if 'insulin' in str(evs): print(k, bb[0], str(evs)[:260]); break
r=bvd(evs); print(type(r), [tuple(x.shape) for x in r] if isinstance(r,(tuple,list)) else tuple(r.shape))
print('mod_events', inspect.signature(mod_events), [x for x in mod_events.__code__.co_consts if isinstance(x,(int,float,str))][:10])
```