The deviation table `DEV` (stay × hour) and the training one-step design `Dtr2` are intact. `Dtr2` holds next glucose `y`, current and earlier glucose `g0..g3`, labs, the tier, and insulin deviations `dev1..dev6` and raw doses `ins1..ins6` at lags 1–6.

The truth coefficients came from `Dtr2`. To get the model's coefficients under the same specification, I need the same design built for tuning stays, joined to the model's predicted means. Those exist in `f3` (factual pass 3) and `RR.pm` (local head on the sel1 states).

My old design-building functions still exist in the kernel, but I can't see their source. So I'll inspect their signatures and the names and constants they use, and reuse them rather than rebuild by guesswork. A mismatch in alignment, such as an off-by-one in lags, would bias Δβ.

```python
import inspect
for fn in ['design_glu','build_ins_design','build_ins_design2','add_dev_lags','regs','regs2','ins_regress','ins_regress2','fe_reg','tier','tier_of','true_shift','deconv','lp','lp_ne','lp_ne2','drug_after12','long_table','last_obs','grid','head_mean','head_p','head_parts','precompute_q','predict','view_of','state_parts','stay_bundles','split_atoms','atom_vec','set_encode','hid_rows','full_mean','f_base','nll_of']:
    f = globals()[fn]
    try:
        sig = str(inspect.signature(f))
    except Exception as ex:
        sig = 'ERR'
    co = f.__code__
    consts = [c for c in co.co_consts if isinstance(c,(str,int,float)) and not (isinstance(c,str) and len(c)>40)]
    print(f"{fn}{sig}\n   names={co.co_names[:25]}\n   consts={consts[:25]}")
```