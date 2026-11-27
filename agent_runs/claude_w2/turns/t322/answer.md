The third factual check also passes. On another 100 tuning stays, edited value NLL is 1.66366 against the original's 1.66225, a change of +0.0014.

| Stay set | Change in value NLL |
|---|---|
| sel1 | −0.0021 |
| sel2 | −0.0002 |
| sel3 | +0.0014 |
| Pooled, 14,406 readings | −0.0004 |

The sets vary by about ±0.002, so the edit is neutral for factual fit. Exceeding the +0.005 limit on the evaluation stays would take a deviation well beyond anything seen here.

The edit is final: in `embed.numeric.weight`, the dose column (z_dose) ×3 and the rate column (z_rate) ×10, nothing else changed. This last Python run records the final state and evidence in the notebook and confirms the saved tensor is what I think it is.
- **Confirms:** the stored column ratios read [1, 1, 3, 1, 10, 1] and only that tensor is edited.

```python
import numpy as np
a=np.asarray(load('embed.numeric.weight'),dtype=np.float64); b=np.asarray(load_original('embed.numeric.weight'),dtype=np.float64)
print('edited:', edited(), 'col ratios', [round(float(a[:,j]@b[:,j]/(b[:,j]@b[:,j])),3) for j in range(a.shape[1])])
with open('/work/notes.md','a') as f:
    f.write('''
## FINAL (run 298) - edit hash eb1bf45c92
- Final edit: embed.numeric.weight col 2 (z_dose) x3, col 4 (z_rate) x10; all else original (head readout cand3 dropped: +0.044 nats, only -1.5 extra insulin effect).
- Tool factual (paired, tuning): sel1 1.52793->1.52579 (-0.0021); sel2 1.62820->1.62797 (-0.0002); sel3 (ids 156-270) 1.66225->1.66366 (+0.0014). Pooled ~-0.0004. Budget +0.005 safe.
- Tool dose_edit sel1 (84 ins / 56 NE stays): orig ins x2 -2.14, x.5 +1.26, NE x2 +1.08, x.5 -0.73 -> edited ins x2 -1.92, x.5 +1.98, NE x2 +3.13, x.5 -2.30.
  sel2 (39/13 stays): orig -4.18/+2.23/+1.26/-0.95 -> edited -4.46/+2.63/+4.08/-2.62.
- Local engine (sel1, validated vs tool within ~0.1): NE x0 -1.66, x3 +4.25, x1.5 +2.09, map_cuff x2 +2.77; insulin x0 +2.12, glucose_lab x2 -1.88.
- Beliefs: record-based effect estimates (one-step OLS lag kernel with glucose lags + policy controls, propagated): insulin x2 ~ -16.5 mg/dL (linear in dose change),
  NE x2 ~ +4.7, x.5 ~ -2.35 mmHg. Model's insulin pathway cannot be strengthened cheaply: dose info enters only via START-atom u-set, decays in recurrent state;
  amplifying encoder dose inputs does not change glucose response; head readouts are entangled with glucose band (factual cost too high).
  NE response saturates (hump) for large amplification: x2 peaks near k4~10 with low k2; x0/x3 under-predicted.
- STATUS: DONE, submit.
''')
print('notes written')
```