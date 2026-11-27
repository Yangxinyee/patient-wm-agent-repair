Run 95 listed the remaining kernel objects. My latest work was checking the encoding of bundle events (e) and actions (u) through the z-posterior: `zpost_from`, `z_check3`/`z_check4` and `zerr_custom`. The function sources cannot be retrieved. Run 93 suggests the action and end-only encodings are now near-exact (relative error about 0.003). Glucose reading bundles still differ slightly (about 0.013 to 0.04). There is no verified core-step or rollout code yet.

The plan stays the same. I will build an exact local copy of the model: encoding, core step, value head and rollout. Then I can test candidate weight edits on any dose intervention cheaply and check them against the tool. The candidate I lean towards is a value-head correction proportional to the dose's deviation from the sliding-scale policy. It fixes the model's under-use of within-band dose variation and also removes the treatment effect absorbed into its glucose dynamics. To implement it exactly I need the precise computation, so first I re-read ARCHITECTURE.md, the first half here. If the core-step description matches my notes (a_bar decay, prelude/blocks/coda), I can code the step directly and check it against the fwd_3 states.

```python
txt=open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:9500])
```