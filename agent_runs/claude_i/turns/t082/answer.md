The core is made of standard parts:
- a prelude: one pre-norm layer with self-attention and an FFN;
- two recurrent layers, each with self-attention, cross-attention to M and an FFN;
- a coda layer (self-attention and FFN);
- the linear maps c_out and b_in, an age MLP, and the per-dimension parameters flow.a, a_in and delta_in;
- out_norm, prelude_norm and slots.

The z networks, gap and heads must sit under other prefixes. Some things are not given by the shapes, chiefly the number of attention heads and the activation. I can settle those by testing candidates against the fwd_3 states, which list out, h and the z mean after every bundle. My old helpers (bundle_eu, zpost_check) cover the encoder and z posterior.

Next I'll list the remaining tensor groups, to see where z_prior, z_post, z_in, gap and the heads live, and how many tensors a full offline step needs. A short, complete list would make re-implementation feasible.

```python
from collections import defaultdict
grp_ = defaultdict(list)
for n in NAMES:
    if n.startswith('core.') or n.startswith('encode_action.'): continue
    p = n.split('.')
    key = p[0] if p[0] not in ('embed','encode','heads','attr_heads') else '.'.join(p[:2])
    grp_[key].append(f"{'.'.join(p[1:])}{list(SHAPES[n])}")
for k, v in grp_.items():
    print(k, len(v), '|', '; '.join(v)[:400])
```