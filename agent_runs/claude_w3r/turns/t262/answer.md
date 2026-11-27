The /work/impl*.py files only hold the data-preparation code (bundling, reading the CSVs, `m_of` for the band mean). The validated model-side helpers (`step_at`, `run_from`, prelude attention, `u_of`, the reading-head features) exist only as kernel objects. I will use them as black boxes, so I need their argument lists.

There is also a design risk to check. Earlier notes say the linear rate read from the normalised action token saturates above about 4. Under the ×2 edit, insulin rates reach about 8, so a saturating read would understate the main readout. Next I print the signatures of the key helpers and the names and shapes of the arrays held in memory. Then I can test the read's linearity at rates 0–10 in the run after.

```python
for k in ['u_of','u_enc','n1u','n1_of','n1e','prelude_att','prel_heads','att_edit','fitread','ridge_read','step_at','core_step_v','run_from','pred_mean_gf','wc_row','head_inputs','dnll','score','targets','mha_w','LNnp']:
    f=globals()[k]; c=f.__code__
    print(k, c.co_varnames[:c.co_argcount], f.__defaults__, '|', ' '.join(c.co_names)[:160])
import numpy as np
print({k:getattr(globals()[k],'shape',None) for k in gg['ndarray'] if k in globals()})
```