The run worked. Three findings:
- **fwd_2:** factual run of the original model on the tuning set. Value NLL per reading is 1.519, the baseline for the 0.005-nat limit.
- **fwd_3:** states on 100 tuning stays, 5,075 bundles.
- **Value head:** a small two-layer network (192 to 192 to 17 bins), plus a lookup on the last value bin.

That gives a workable plan. Compute the head's input locally from the returned states, find a direction that carries recent dose information, and add a controlled tilt to the glucose and MAP value logits. Then calibrate its strength with dose_edit passes.

First I need my helper functions from earlier runs: the policy model and the truth calculation. I'll print their signatures and the remaining embedding and encoder shapes, keeping the output short.

```python
import inspect
for fn in ['panel','base_ins','ctrl_ins','ctrl_ne','lp','deconv','true_shift','lin','LN','mlp','encode_set','atom_vec']:
    src=inspect.getsource(globals()[fn]).splitlines()
    print(fn, '|', src[0][:110], '| lines', len(src))
keys=['embed.bins','embed.categorical','encode_action','z_prior','gap.','last_numeric','type_emb','embed.code','last_status','z_post','z_in']
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES if any(n.startswith(q) for q in keys)))
print(list(VOCAB.keys()), VOCAB.get('config'))
```