The heads section settles it. The glucose value head reads only the state view plus a register of the last value of the same code: its value, age, bin and a transition table. No head input carries the dose, so the edit has to make the state update carry it.

All dose information enters the state through just two prelude tokens: u (the encoded drug start) and z_in(z_post). Only z_post sees both the context (glucose level, hence policy-expected rate) and u (actual rate). It can therefore compute a gated deviation (rate − expected rate) in spare hidden units without disturbing factual fit. The open question is whether the pre-norm prelude lets a z-token change move the next glucose prediction by several mg/dL while staying roughly linear.

Next I'll recover the call signatures of my existing helpers and check the scale of the saved Jacobians JI/JN (units and magnitude per z dim). That tells me whether the existing z_in columns are useless or just weak.

```python
import inspect
for fn in ['core_step','step_err','step_local','stepk','run_stay_z','jac_at','pred_mean_s','value_mean','view_ctx','zprior','zprior2','pibar','tgt_dev','true_shift','dev_ins','eu2','zpost2','emb_atoms','set_enc2','u_for','head_value2','vm_custom']:
    try: print(fn, str(inspect.signature(globals()[fn]))[:150])
    except Exception as ex: print(fn, 'ERR', type(ex).__name__)
print('JI |.| mean per out:', np.abs(JI).mean((0,1)).round(4), ' max dim:', np.abs(JI).mean((0,2)).argmax(), np.abs(JI).mean((0,2)).max().round(4))
print('JN |.| mean per out:', np.abs(JN).mean((0,1)).round(4))
```