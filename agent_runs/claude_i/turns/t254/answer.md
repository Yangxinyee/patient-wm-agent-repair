The kernel still has the full toolkit: `core_step`, `step_err`, `run_stay`, `run_stay_z`, `zpost2`, `eu2`, `view_ctx`, `zprior`, `pibar`, `value_mean` and `tgt_dev`. My earlier session apparently also wrote stay-level simulators (`run_stay` and `run_stay_z`). The array listing failed because a numpy function has a `.shape` attribute.

Next I'll list only real arrays and tensors, plus the function definitions in the /work .py files. That tells me which globals hold the fwd_3 states and the weights. If arrays like HHt, OUTt, Ee, Uu and ZZt are present, I can go straight to designing the z_post edit.

```python
import numpy as np, torch, re, inspect
arrs = sorted((k, tuple(v.shape)) for k, v in globals().items() if not k.startswith('_') and isinstance(v,(np.ndarray, torch.Tensor)))
print(len(arrs)); print('; '.join(f"{k}{s}" for k, s in arrs)[:1800])
oth = sorted((k, type(v).__name__, len(v)) for k, v in globals().items() if not k.startswith('_') and isinstance(v,(dict,list)) and k not in ('In','Out'))
print('; '.join(f"{k}:{t}{n}" for k,t,n in oth)[:600])
for f in ['sim.py','jac2.py','enc2.py','head2.py']:
    s = open('/work/'+f).read()
    print(f, len(s), re.findall(r'\ndef (\w+)\(', '\n'+s)[:14])
```