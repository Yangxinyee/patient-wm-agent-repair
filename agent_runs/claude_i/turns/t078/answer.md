On the 100 tuning stays the gap is large:

| Edit | Kernel-implied true shift | Original model |
|---|---|---|
| insulin x2 | -12.5 | -1.85 |
| insulin x1/2 | +6.2 | +1.07 |
| norepinephrine x2 | +4.3 | +0.36 |
| norepinephrine x1/2 | -2.2 | -0.06 |

So the original model responds about 6 to 12 times too weakly. Next I'll check whether that weakness also shows in teacher-forced prediction or only in rollouts. I will print the regression results I left in res, r_, bb and se, and record these targets in the notebook. A teacher-forced dose coefficient well below the data's would mean the model under-learned the per-step effect. That would make a stronger, correct dose effect improve factual NLL rather than harm it.

```python
print(res.round(3).to_string()); print(r_.round(3).to_string()); print(bb, se)
with open('/work/notes.md','a') as f:
    f.write("""
## Session 2 (restart after reply 70)
- fwd_2 = factual original (100 stays), fwd_3 = states original. Built numpy head reproduction from states: mean MAD ~1.3 vs tool (prior-z sampling noise).
- TARGETS on sel1 (kernel-implied true, mean over reached stay-hours): ins x2 -12.5, ins x1/2 +6.25, ne x2 +4.33, ne x1/2 -2.17.
  Original model: -1.85, +1.07, +0.355, -0.06. Model is ~6-12x too weak.
""")
```