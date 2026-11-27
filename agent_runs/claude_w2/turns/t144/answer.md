The function sources were not saved, so I can't recover the exact head and core code from runs 120-133. What the stored results do show:

- The head reproduction reaches correlation 0.9989 with the tool's predicted means, but with RMSE ~2–3 mg/dL. That residual is plausibly z-sampling noise: the tool draws z from the prior 4 times and averages.
- The core step was being tested with several age-encoding variants (`res_`, best ~3.8).

From now on I'll save every working function to /work files so restarts don't lose code. The next step is to list the core/head tensor names and shapes. Then I can write a clean, documented core step and test it bundle by bundle against the exact states from fwd_3. An exact match would allow local counterfactual rollouts. A persistent mismatch would point to the age or memory definition.

```python
core=[(n,SHAPES[n]) for n in NAMES if n.startswith('core') or n.startswith('gap') or n.startswith('z_') ]
print(core)
others=sorted(set(n.split('.')[0] for n in NAMES)); print(others)
print({k:V['config'][k] for k in V['config']})
print(sorted(res_, key=lambda r:r[0])[:8])
print(S.files, {f:S[f].shape for f in S.files})
```