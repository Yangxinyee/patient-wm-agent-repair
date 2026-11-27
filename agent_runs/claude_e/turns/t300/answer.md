With tu1 rerun under the exact saved edit (x2 -12.85, x1/2 +7.51, NE unchanged), all four tuning batches are measured at -2.7. Weighting by reached stay-hours (2,009 for insulin, 1,080 for NE):

| Readout | Original | Edited (-2.7) | Record-based truth | Edited / truth |
|---|---|---|---|---|
| Insulin x2 → glucose_finger | -1.80 | -12.76 | -13.55 | .94 |
| Insulin x1/2 → glucose_finger | +0.82 | +6.05 | +6.77 | .89 |
| NE x2 → map_arterial | -0.23 | -0.23 | +4.53 | — |
| NE x1/2 → map_arterial | -0.56 | -0.56 | -2.26 | — |

The step from -2.6 to -2.7 raised both insulin readouts by about 3.5%, as predicted. Going further, to -2.8 or -2.9, would bring them closer, but it roughly doubles the risk of breaking the .005 factual limit if unseen stays behave like the tuning batches. The expected net gain is about zero, so I will keep -2.7.

The NE readouts are untouched. The only head-level NE fix I found costs far more factual fit than the budget allows, and NE x2 needs rates the model never saw.

I will record the final state and evidence in the notebook, and recompute the pooled table from the files to make sure the numbers above are exact. Then I will submit.

```python
summ_files = {'f3': 36, 'tu0': 37, 'tu2': 38, 'tu1': 39}
rows = []
for lab, k in summ_files.items():
    for m in ['original', 'edited']:
        d = pd.read_csv(f'/work/fwd_{k}/dose_edit_{m}.csv'); d = d[d.reached]
        for e, g in d.groupby('edit'): rows.append((lab, m, e, len(g), g['shift'].sum()))
T = pd.DataFrame(rows, columns=['batch','model','edit','n','s'])
P = T.groupby(['model','edit'])[['n','s']].sum(); P['mean'] = P.s / P.n
truth = {'insulin_x2': -13.545, 'insulin_half': 6.773, 'ne_x2': 4.528, 'ne_half': -2.264}
out = P['mean'].unstack(0); out['truth'] = [truth[e] for e in out.index]; out['ratio_edited'] = out['edited']/out['truth']
print(out.round(3).to_string())
with open('/work/notes.md','a') as f:
    f.write("\n## Runs 252-258 (restart 7, FINAL)\n- Kernel restarted (OOM) at run 253; helper functions lost; saved edit on disk intact = ins_only(-2.7) (W0/b0 rows 26/41, W2 cols 26/41 = +-v, norm 3.83, scale with gain).\n"
            "- Factual cost at -2.7 (direct or gain^2-scaled), 14 batches x100: tuning .0046 .0043 .0033 .0011; old train .0018 .0028 .0040 .0011; fresh train (fwd_30-35) .0027 .0032 .0010 .0002 .0026 .0028. Pooled mean ~.0025, tuning-only .0033, batch sd .0013.\n"
            "- Bootstrap P(>.005) at -2.7: all-pool N=400 .2%; tuning-only N=400 1.45% (+mean uncertainty -> few %). Kept -2.7 (higher gain ~ same expected value, more risk).\n"
            "- dose_edit at -2.7 on all 4 tuning batches (fwd_36-39) pooled:\n" + out.round(3).to_string() + "\n- FINAL: ins_only(-2.7), hash 2bc751d207. NE not edited.\n")
```