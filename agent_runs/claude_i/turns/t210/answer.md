Output has to stay short, so from now on I will print only summaries.

What run 193 showed:
- `z_post` has 49 dead hidden units. Their pre-activations average −4 to −7, so their output is essentially zero. I can rewrite their input and output weights without changing any current prediction. Units 1, 6, 11 and so on are free.
- The old Jacobian arrays (`mJ`, `sJ`) give consistent signs: two of the four outputs fall by about 0.05–0.24 per unit of z in almost every direction, while the other two barely move. I don't know which outputs those are or how they were computed, so I will redo the computation in my own saved code, not rely on them.

Next I need the layout of the per-bundle metadata: the atom lists for a drug bundle and a reading bundle, the readings table `meas`, and the verified `core_step` configuration. With that I can compute how a change in z at an insulin bundle moves the next predicted glucose and MAP.

This run only collects layouts. It is useful if `atoms_by_bundle` gives code and value per atom, so I can rebuild the glucose register (last value and its age) for each reading.

```python
print('V', V, '| V_ok', V_ok)
kd_ = int(drugb[5]); print('drug bundle', kd_, 'sid', sid3[kd_], 'hr', hr3[kd_])
print(str(atoms_by_bundle[kd_])[:300])
print(str(atoms_by_bundle[kd_+1])[:300])
print('meas cols', list(meas.columns)); print(meas.head(2).to_string()[:400])
print('tmin?', 'tmin' in globals(), '| first_of', first_of[:5], '| pk[:3]', pk[:3])
```