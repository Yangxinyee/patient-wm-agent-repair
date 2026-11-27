The injection code shows how far I had got. `bstep_z` swaps the z token for a custom function `zfn(ctx, e, u, info)`, where `info` holds the exact insulin and NE rates, and `tf_stay_z` scores teacher-forced NLL with that function. This is the Stage-1 prototype, and no weight edit for it has been saved.

The kernel's own function sources are lost, so I'll rebuild compactly around the parts I can read from files. The quickest route is to put the residual-dose injection directly into the z_post and z_in weights. Then the existing exact replicas (teacher-forced and rollout) test the real weight edit with no `zfn` hook.

First I need the rollout replica's interface (`rollout(W, B_, conds, npaths, seed)`) and how its step builds z. That tells me whether an edited W automatically flows through.

```python
src = open('/work/wm2_rollout.py').read(); print(src[:4300])
s2 = open('/work/wm2.py').read(); i = s2.find('def bstep'); print(s2[i:i+1500])
```