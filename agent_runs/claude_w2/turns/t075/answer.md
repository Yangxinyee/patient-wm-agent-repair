Bin side makes no difference, but measuring the age of the last reading in minutes instead of hours does help: error falls to 2.34 and NLL to 1.4705, against the tool's 1.4235. The register inputs are therefore still not fully right, and I had fixed their layout earlier under the wrong activation (SiLU).

Next I redo the greedy column-by-column search over the 8 `last_numeric` inputs with GELU and left-sided bins. Candidate features are has_last, no-previous-reading, log-age in seconds, minutes or hours, the last value's z, its square, and constants. I start from the current best layout.

- **Good outcome:** a sensible layout with error below 1.5 and NLL near 1.42.
- **Bad outcome:** a nonsensical layout with little gain. That would point elsewhere, for example to the state itself, and I would settle for the current fit, which is already good for finger glucose.

```python
feats2 = {'zero': np.zeros(len(fa4)), 'one': np.ones(len(fa4)), 'has': fa4.has.values, 'nohas': 1-fa4.has.values,
          'la_s': np.log1p(fa4.age_s.values)/10*fa4.has.values, 'la_m': np.log1p(fa4.age_s.values/60)/10*fa4.has.values,
          'la_h': np.log1p(fa4.age_s.values/3600)/10*fa4.has.values, 'zl': fa4.zl.values, 'zl2': fa4.zl.values**2/10}
asg2 = ['has','la_m','zl','has','zero','zero','zero','zero']
def ev2(asg):
    pmv, _, nlv, _, _, _ = run13(np.stack([feats2[a_] for a_ in asg],1), lbL, obL); return (round(float(nlv.mean()),4), round(float(np.abs(pmv-tgt).mean()),3))
cur = ev2(asg2); print('start', cur)
for it_ in range(2):
    for c_ in range(8):
        best_c = (cur, asg2[c_])
        for fk in feats2:
            a2 = asg2.copy(); a2[c_] = fk; r_ = ev2(a2)
            if r_[0] < best_c[0][0] - 1e-4: best_c = (r_, fk)
        asg2[c_] = best_c[1]; cur = best_c[0]
    print(it_, asg2, cur)
```