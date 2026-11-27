The z-token probe shows a dead end. The response along the best glucose direction is the same at norms 1, 3, 10 and 30, and it is tiny: −0.02 mg/dL on glucose, +0.06 on MAP. Even replacing the z token's direction outright moves the next glucose prediction by well under 1 mg/dL. Per-bundle directions are also inconsistent (median cosine 0.14).

So the z token is pre-normalised and the slots barely attend to it. Editing z_post and z_in alone cannot inject a dose effect of several mg/dL. Dose information must get into the slots through an amplified attention path, and I need to know how the prelude attention is built to design that.

Next I'll list the prelude and z_in tensor shapes (number of heads) and the signature of my existing `attnw` helper. With multiple heads, one rarely used head could be dedicated to carrying the deviation from the z token into the slots.

```python
for n in NAMES:
    if n.startswith('core.prelude') or n.startswith('core.z_in') or n in ('core.slots','core.b_in.weight','core.delta_in','core.a_in'):
        print(n, SHAPES[n])
for fn in ['attnw','act_','actf','actfn','hidden_of','head_hidden','u_from_V','eu_routed2']:
    try: print(fn, str(inspect.signature(globals()[fn]))[:120])
    except Exception as ex: print(fn, 'ERR')
```