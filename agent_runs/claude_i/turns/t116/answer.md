Printing the helper files was too long and the output was blocked, so I will keep outputs short from now on.

The architecture review gave me what I need to plan. The glucose value head reads the drugs only through the state that the drug bundle vectors u_k leave behind. So a "what if" dose change works only through how the state responds to the rate and dose features.

The central question for the edit is what the original model does with dose variation inside the range seen in training. There are two possibilities:
- **It already uses that variation correctly** and only extrapolates badly to doubled doses. Then an edit must leave factual predictions alone and fix only the off-range response.
- **It largely ignores the dose noise.** Then adding the true drug effect would also improve the factual fit.

My kernel holds `F2`, `X_`, `bb`, `rr` and `se`, which look like a residual-on-dose regression I ran on the factual pass (fwd_2) just before the restart. The notebook does not record its result. I will print a compact summary of those variables and add a session-3 header to the notebook.

How to read it: coefficients of the factual residual on recent insulin doses near the true kernel (-2 per u/h at lags 1 to 3) mean the model ignores dose noise. Coefficients near 0 mean it already captures the within-range effect.

```python
import numpy as np, pandas as pd
print('F2', F2.shape, list(F2.columns)[:30])
for nm_ in ['X_','bb','rr','se']:
    v_ = globals()[nm_]
    print(nm_, getattr(v_,'shape',None), np.round(np.asarray(v_).ravel()[:24],3))
print('K_ins', np.round(K_ins,2)); print('K_ne', np.round(K_ne,2))
with open('/work/notes.md','a') as f_: f_.write('\n## Session 3 (restart after reply 110)\n- Re-read ARCHITECTURE. Plan: test whether model uses within-support dose noise (residual regression), then design edit.\n')
```