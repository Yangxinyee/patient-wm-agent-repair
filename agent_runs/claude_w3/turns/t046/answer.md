I re-read the architecture, and my kernel variables are still there: theta, th_ins, th_ne, the panels and sel. The only route from a dose to a later glucose or MAP prediction is this chain:
- the action atoms (dose and rate z-scores and bins) feed `encode_action`, which gives u_k;
- u_k feeds z_post and the prelude;
- these update the state h and `out`;
- the heads read out the prediction.

In the rollout, the predicted readings are sampled and fed back through the "last value" register. So any shift the edit introduces will carry forward through time.

An edit that simply adds a full dose effect would double-count. The original model already captures the average post-insulin fall through the glucose level, because dose ≈ base(G) × jitter. The edit therefore needs to respond to the dose relative to what the state already implies.

Next I want a compact, short-output view of the tensors: names grouped by module, with shapes, for the action encoder, the prelude inputs and the value head. I also want to see what fwd_2 contains. That tells me whether to edit the action encoder, the prelude or the head, and whether I can rebuild the head (or the whole step) locally and check it against the fwd_3 states.

```python
import json, collections
T = json.load(open('/model/tensors.json'))
grp = collections.OrderedDict()
for nm in NAMES:
    p = nm.split('.')
    key = '.'.join(p[:2]) if p[0] not in ('core','embed','attr_heads') else '.'.join(p[:3])
    grp.setdefault(key, []).append(nm)
for k_, v_ in grp.items():
    print(k_, len(v_), sum(int(np.prod(SHAPES[x])) for x in v_))
print([ (x, SHAPES[x]) for x in NAMES if x.startswith('encode_action') ][:20])
print(os.listdir('/work/fwd_2'), os.listdir('/work/fwd_1'))
```