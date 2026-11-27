The architecture text clears up the main design questions.

- **How doses reach the readings.** Doses affect the predicted readings only through the state, which then feeds one value head shared by all reading codes.
- **What makes the correction code-specific.** The only code-specific inputs to that head are the code embedding, the "last value" register and a bias table indexed by code and last value bin.
- **What the readout is.** It is the predicted mean of a 17-class softmax.

So the edit I'm pursuing is a correction inside the value head, gated by reading code. It would shift glucose predictions in proportion to a probe of the insulin dose residual read from q, and MAP predictions in proportion to a probe of the norepinephrine dose residual. Everything will be computed analytically from regressions on states and records, with a few gains then calibrated by dose_edit. No gradient training and no optimiser over the model's weights.

My helper functions have no retrievable source, so I first need to confirm what the stored arrays hold:
- Q99 / WC99 / PRE / TR99 / CENm / pmall, covering 5,521 readings of the 99 state stays.
- Whether a fresh, transparent rebuild of the value head from WC99 reproduces the tool's predicted means and per-reading NLLs from forward pass 4.

If my means match the tool's closely, I can trust offline checks of what an edit costs factually. If they don't match, I'll rebuild q before going further. I'm also logging the restart state in the notebook now.

```python
import pandas as pd, numpy as np, json
from scipy.special import erf
open('/work/notes.md','a').write("""
## Restart 2 (after run 121; runs 74-121 were NOT logged - recovered from kernel vars)
- fwd5-7 = states(original) for new300.json stays (3x100) -> S5,S6,S7 / D5,D6,D7. errs[300]: some recon check, mean err .056.
- Kernel arrays for 99 sel_states stays (5521 readings): Q99 (q), WC99 (with_code), PRE (value-head pre-acts), TR99 (transition bias), CENm (centres), pmall; eps_ins/eps_ne (30000x25) dose residuals per stay-hour.
- No function source retrievable; rebuild key pieces explicitly.
- PLAN: paired GELU units in value head (A = B + eps*probe, W2 cols +-lam*tilt; gate by code-direction), removed units' mean absorbed into embed_transition.value_bin; probe on q for dose-shock part of factual error; calibrate gains via dose_edit per-hour kernels. Everything analytic/regression-based, no optimiser over weights.
""")
vn=[n for n in NAMES if any(k in n for k in ['value_bin','query_norm','last_numeric','type_emb','last_status'])]
print([(n,SHAPES[n]) for n in vn])
fa=pd.read_csv('/work/fwd_4/factual_original.csv'); print(fa.columns.tolist(), len(fa)); print(fa.head(2).to_string())
print('R99 rows:\n', np.round(R99[:2],3))
print('VOCAB keys', list(VOCAB.keys()), type(VOCAB['value_bin_centres']) if 'value_bin_centres' in VOCAB else '')
```