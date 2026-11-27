Run 206 shows the slot state is not a clean memory. `c_out` is a generic mixing matrix far from the identity, and the per-step decay `a_bar` is about 0.35. The largest eigenvalue of the linear recurrence is about 0.56, so a linear write into `h` fades within one or two bundles. Persistence comes from the model's own nonlinear machinery: prelude, attention blocks, coda.

**New plan: write into the z posterior along the "observation surprise" direction.** The model already carries z into its beliefs through that machinery, and readout weights near 1 mean nothing extra strips it on the way out.
- **Directions:** estimate, from the states I already have, the direction in z that the model uses to encode a glucose surprise and a MAP surprise (observed minus predicted).
- **Writer:** repurposed dead z_post hidden units, gated to drug bundles. Each computes the dose innovation, the drug rate minus the rate the state predicts, and adds a scaled copy of that surprise direction.
- **Why innovations:** they keep the factual predictions consistent and should not add bias.

First I need the rest of my kernel inventory: the states, reading-to-row maps and factual outputs from earlier runs.

```python
import types
skip=set(['In','Out','exit','quit','get_ipython'])
arr=[];oth=[]
for k,v in list(globals().items()):
    if k.startswith('_') or k in skip or isinstance(v,(types.ModuleType,type)): continue
    if isinstance(v,(np.ndarray, torch.Tensor)): arr.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v, types.FunctionType): arr.append(f"{k}()")
    else: oth.append(f"{k}:{type(v).__name__}")
s=' '.join(arr); i=s.find('stays_arr'); print(s[i:i+2000]); print(' '.join(oth)[:1200])
```