The inventory shows the pipeline is in the kernel: `fit_eval`, `oof_fit`, `oof_units`, `frontier`, `amp_eval`, `set_amp`, `solve_theta`, `tilt_mean`, `ridge`, `run_cf` and `gen_stay`. The source files hold only parts of it. Run 207's output probably came from `fit_eval` or `frontier`.

Next I will read the signatures and the global names these functions use, which works without the source. That should tell me what "expF" is and how the edit is built and applied. If the signatures and globals map cleanly onto the steps in my notes (ridge readout on WC, then tilt, then unit edit), I can reuse them. If not, I will rebuild the fit step in a file.

```python
import inspect
for n in ['fit_eval','frontier','oof_fit','oof_units','amp_eval','set_amp','solve_theta','tilt_mean','ridge','run_cf','gen_stay','heads_full','true_shift','rows_truth','add_targets','fact_nll','dnll_groups','cf_stay','run_variant3','with_code','value_probs']:
    f=globals()[n]; c=f.__code__
    print(n,str(inspect.signature(f))[:80],'| G:',','.join([x for x in c.co_names if x in globals()][:18])[:160])
```