The last run showed my kernel already holds more helpers than the notebook records, including `core_step`, `check`, `step_err`, `view_ctx`, `zprior`, `pred_mean`, `pm_both`, `policy_tables` and `pibar`. So in session 3 I seem to have written and maybe checked a full core-step reproduction without noting it.

The core is pre-norm attention layers, and c_out is a full 192×192 linear map before the coda. No dimension of h passes through a step untouched, so a "free" persistent dose-memory channel is unlikely.

Before building anything new I want to recover what these functions do and how accurate they were. I will print their source with `inspect` and write the missing facts into the notebook. If the source shows `core_step` reproducing the tool's states to a small error, I can simulate dose edits locally. If `inspect` cannot recover the source, I will have to rewrite and re-check it.

```python
import inspect
for nm_ in ['core_step','check','step_err','eu_with_end','end_embed','view_ctx','zprior','pred_mean','pred_mean_v','probs_det','pm_both','policy_tables','pibar','attnw','ffnw','age_emb','LNw','linw']:
    try:
        print("#####", nm_); print(inspect.getsource(globals()[nm_]))
    except Exception as ex:
        print("#####", nm_, "FAILED", ex)
```