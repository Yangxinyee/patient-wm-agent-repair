The listing confirms the offline machinery is still in the kernel:

- the core and encoder replicas (`core_step_b`, `encode_bundles`, `make_atoms`);
- the value head (`head_value`);
- the teacher-forced replica (`tf_stay`, `tf_predict`) and the rollout replicas (`rollout_stay`, `rollout_gen`);
- the record-side effect estimates (`psi_ins`, `psi_ne`, `true_shift`).

So I can test candidate edits offline before spending forward passes. The output was cut at 10,000 characters before the architecture document printed. Since I need the exact update and head equations to design a dose-memory channel, I will print ARCHITECTURE.md now, in pieces if needed.

The result I am looking for is a description of how h is updated (`abar`, `sp`, `b_in`, `flow.a`) and how the action vector u is formed from the infusion atoms. That would tell me whether a near-linear write path for the rate exists.

```python
A_ = open('/model/ARCHITECTURE.md').read()
print(len(A_))
print(A_[:9500])
```