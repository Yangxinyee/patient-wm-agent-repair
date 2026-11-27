Runs 243 and 244 showed that the helpers built in runs 177–240 have no recoverable source. They were defined in the kernel, so `inspect` cannot read them. Only the argument names are left. For example, `rollout(W, B_, conds, npaths, seed, t_edit, t_hide, nz)` and `true_shift(drug, si_, t_, fac, start, irf)`. One of them, `make_zfn(ki, kn)`, looks like a Python-level test function that injects residual signals with two gains, insulin and norepinephrine. It is not a weight edit.

The rollout replica, though, was saved as a file: /work/wm2_rollout.py. I need its exact code before any offline calibration. It tells me how a custom z-token function plugs into the rollout and how the dose conditions are written. It also tells me how the readout is pooled, so that offline numbers match the tool. I'll print that file, plus the small control objects (`conds`, `FAC`, the stored effect estimates `IRF_I`/`IRF_N`).

Expected outcome: the rollout calls `bstep_z` with a z-token function, taken either as an argument or from a global. If so, I can test a weight-based z_post/z_in edit offline directly. If it cannot take one, I'll add that hook first.

```python
s = open('/work/wm2_rollout.py').read()
print(len(s)); print(s)
print('conds', conds, '| conds4', conds4, '| FAC', FAC)
print('IRF_I', np.round(IRF_I, 3), '| IRF_N', np.round(IRF_N, 3))
print('b1 len', len(b1), b1[:3])
```