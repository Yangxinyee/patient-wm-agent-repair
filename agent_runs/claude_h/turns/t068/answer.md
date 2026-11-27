Pass 3 was the dose_edit check (100 tuning stays: 39 with insulin, 5 with norepinephrine in the window). The edited model's shift is exactly 0.0 at every stay-hour for all four edits. The original model on the same stays gave insulin x2 +0.052, x1/2 −0.152, NE x2 +0.008, NE x1/2 −0.023, again essentially 0.

So far the edit does what it should: dose-invariant readouts and an unchanged factual fit. But the architecture file mentions a second route by which a drug rate enters the model. It is a "last value" feature, `last_numeric([..., z_last_rate])` plus `embed.bins.rate[last rate bin]`, and I have only neutralised its bin part.

The zero shifts suggest this route does not reach reading predictions. Pass 3 had only 5 norepinephrine stays, though, so I want to read the architecture and know rather than infer. The hypothesis "the edit is exactly dose-invariant for any stay" is confirmed if `z_last_rate` feeds only the action-prediction heads, which the rollout does not use because doses are given. It is refuted if it feeds the reading heads or the state. I'll also log passes 3 and 4 in the notebook.

```python
lines=open('/model/ARCHITECTURE.md').read().splitlines()
print('\n'.join(f'{i}: {l}' for i,l in enumerate(lines) if 24<=i<=100))
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 55-63 (after restart 2)
- FWD 3 (dose_edit both, tuning 100 stays; 39 insulin stays/397 stay-hours, 5 NE stays/56 stay-hours): original ins x2 +0.052, x1/2 -0.152,
  ne x2 +0.008, ne x1/2 -0.023; EDITED all shifts exactly 0.0 at every stay-hour (shift_se 0).
- FWD 4 (factual both, 100 other tuning stays, 4529 values): original 1.58471, edited 1.58464 (-0.00007). NLL SAFE (2 checks).
- Checking ARCHITECTURE for other dose paths (last_numeric z_last_rate, last rate bin) before finalizing.
""")
```