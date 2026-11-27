My new `zparts` reproduces the recorded z exactly (median and max error 0.0000). The posterior mean is just the first 32 outputs of `z_post([ctx, e, u])`. The run-141 mismatch was a bug in my old `zcheck2`, not in the model. So ctx, e, u and z are now all exact.

Next I need the rest of the step (prelude, the recurrent block with memory cross-attention, coda) to reproduce h and out. With that I can simulate rollouts locally and design the edit offline. I'll list the core tensor names to check the layer structure against my `self_layer`/`core_step` code (norm names, FFN names, age MLP). If they match, I'll test the full step next. If not, the names show what to fix.

```python
cn = [n for n in NAMES if n.startswith('core.')]
import collections
grp_ = collections.OrderedDict()
for n in cn:
    p = n.split('.'); key='.'.join(p[:3]) if p[1] in ('prelude','blocks','coda') else '.'.join(p[:2])
    grp_.setdefault(key, []).append('.'.join(p[3:]) if p[1] in ('prelude','blocks','coda') else '.'.join(p[2:])+str(tuple(SHAPES[n])))
for k_,v_ in grp_.items(): print(k_, v_)
```