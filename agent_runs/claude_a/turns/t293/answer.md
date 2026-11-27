`rollout_gen` returns, for each hidden reading after hour 12, a 3-element tuple (19 for this stay). It also returns the time each drug first ran after hour 12: insulin at 15.08 h, norepinephrine at 12.08 h. That time defines which stay-hours count as reached. `mult` uses the names insulin_x2, insulin_half, norepinephrine_x2 and norepinephrine_half. I'll print two of the tuples to see what each element holds, presumably the code, the time and the per-spec predicted values. Then I can write the pooled-shift calculation.

```python
for i in (0,5):
    print([ (type(x).__name__, getattr(x,'shape',None), (x if np.ndim(x)==0 else np.round(np.asarray(x,dtype=float).ravel()[:6],2))) for x in L0[i]])
```