Most of the needed data is already in the kernel: the full training record ('tr', about 2.0M rows), an hourly panel ('PT', 'PT_tr') and per-drug record tables ('ins', 'ne'). The insulin-surprise column is nonzero for about 9.3k readings, with sd 0.70. The mean-gradient norms differ by code family (glucose 21, MAP 6.3), as expected from the different value scales.

I still need a norepinephrine surprise per MAP reading, analogous to the insulin one. Earlier runs built a clean version: rate minus its expected value given the expected-rate bin and MAP band. Before rebuilding it, I check the columns of the panel tables to see whether it is already stored.

```python
for k in ['PT','PT_tr','lag','key','ne','ins','tr']:
    print(k, list(globals()[k].columns))
```