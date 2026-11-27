cf.py shows that I had already written a structural direct-effect calculator. It assumes the next reading, given the last same-code reading, carries the dose effect through a pool that decays at 0.6 for insulin and 0.39 for NE. It also has a head evaluator on stored states. The kernel also holds run_cf (counterfactual states from a given minute), mod_events, true_shift, solve_theta/tilt_mean (the logit tilt needed for a target mean), and set_amp/amp_eval/oof_units/fact_nll. That suggests I had already fitted readout "units" and evaluated factual NLL in the undocumented runs.

Before redoing anything I need the stored data objects (counterfactual states, reading tables, fitted weights). Next I list the non-function kernel variables with their types and shapes in compact form.
- If I find stored counterfactual states plus fitted unit weights, I can go straight to building the weight edit.
- If not, I regenerate them with run_cf.

```python
import numpy as np, torch, pandas as pd
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,(types.FunctionType,types.ModuleType,type)): continue
    if isinstance(v,(np.ndarray,torch.Tensor)): out.append(f"{k}:{tuple(v.shape)}")
    elif isinstance(v,pd.DataFrame): out.append(f"{k}:df{v.shape}")
    elif isinstance(v,dict): out.append(f"{k}:d{len(v)}")
    elif isinstance(v,(list,tuple)): out.append(f"{k}:l{len(v)}")
    elif isinstance(v,(int,float,str)): out.append(f"{k}={str(v)[:20]}")
print(len(out)); print(' '.join(out)[:3500])
```