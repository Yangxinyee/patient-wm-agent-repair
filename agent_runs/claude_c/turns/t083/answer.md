The last run re-displayed the model's architecture. It also confirmed that my kernel variables are still alive.

The architecture narrows down how dose can reach later glucose and MAP predictions. Drug atoms are encoded separately into an action vector u_k. That vector enters the state through only two routes: the prelude attention, where it is one token among several, and the z posterior. I already ruled out the z route (pass 8 changed nothing). Earlier input-side edits also gained little, likely because u_k passes through layer norms that discard its overall scale.

So the remaining lever must sit either inside the action encoder or wherever the u-token is consumed in the prelude. Before choosing one, I'll list the exact names and shapes of the tensors on that route: `encode_action`, the prelude layer, `b_in`, and the value head. I'm looking for tensors that only action bundles pass through, because editing those cannot disturb predictions at bundles without drugs. The listing will show whether such dedicated tensors exist, or whether every edit has to go through shared tensors.

```python
for nm_ in NAMES:
    if any(nm_.startswith(p) for p in ['encode_action','core.prelude','core.b_in','core.delta_in','core.a_in','core.slots','attr_heads.value','embed.numeric','embed.bins','core.z_in','core.z_post','embed_transition','last_numeric','core.c_out','core.coda']):
        print(nm_, SHAPES[nm_])
print(len(NAMES))
print([x for x in NAMES if not any(x.startswith(p) for p in ['encode','core','attr_heads','embed'])])
```