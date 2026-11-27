The kernel has a full set of building blocks: `stay_bundles(stay)`, `enc_bundle(group)`, `LN`, `mha`, `mlp`, and the time-flow helper `HN(prev, dt)`. I'll check what `stay_bundles` and `enc_bundle` return for the first test stay. Then I'll use them in a clean check of the z posterior mean, rebuilt from the previous saved state against the saved z. Agreement there would mean the encoders and the context term are correct, so any remaining 6% error is in the recurrent block or the coda.

```python
s0_ = int(F4['stay_id'][0]); B_ = stay_bundles(s0_)
print(s0_, type(B_), len(B_), (F4['stay_id']==s0_).sum())
o_ = enc_bundle(B_[0]) if isinstance(B_, list) else None
print(type(o_), [getattr(t,'shape',type(t)) for t in (o_ if isinstance(o_, (tuple,list)) else [o_])])
```