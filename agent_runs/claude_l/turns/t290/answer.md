Run 266 showed that the `Cdir_*` arrays are 192-dimensional context directions, not lag coefficients. I cannot reliably reconstruct the undocumented objects from runs 177–263, so I will rebuild the Stage-1 edit from my verified primitives:
- the exact replica,
- `bstep_z` / `tf_stay_z`,
- the rollout replica,
- the teacher-forced train data.

To do that I need the exact signatures of the rollout and teacher-forcing code. Next I will print those two short source files and check which weight and bundle caches exist in the kernel. This is a lookup step, so it tests no hypothesis.

```python
print(open('/work/wm2_rollout.py').read())
print(open('/work/tf.py').read())
print([k for k in g if k in ('W','W0','WE','Wt','BUN','TRB','dmap','X0row','u_empty','MEASC','CENd','split_fix','set_enc','build_bundles','batches','B1','tun','TUN')])
```