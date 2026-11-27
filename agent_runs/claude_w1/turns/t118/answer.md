The rows from my multi-stay call came out ordered by hour across different stays. The vectors also fail to predict even the current reading, although they did so with R² 0.90 in `samp_pert`. So `gen_samples` evidently treats its whole input DataFrame as a single record. Feeding it 3,000 stays at once merged them into one nonsense timeline.

Runs 105 and 107 are therefore invalid, including the "head-only is infeasible" conclusion from the decoder noise floor. `samp_pert` must have been built by calling it per stay. I'll redo the sample generation with one call per stay. First I'll time it on 60 stays and confirm that the vectors again predict the current reading.

```python
grp_tr = {s: g_ for s, g_ in trs.groupby('stay_id', sort=False)}
t0 = time.time(); outF = []
for s in st3[:60]:
    outF += gen_samples(grp_tr[s], pert=False)
t1 = time.time()
print('per-stay calls: %.3f s/stay, rows %d' % ((t1-t0)/60, len(outF)))
print('first rows stay/hour:', [(r[1], r[2]) for r in outF[:4]])
```