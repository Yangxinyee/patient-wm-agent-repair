The readout is in place and the factual fit holds: +0.00025 on sel2 and +0.00001 on dsE. Average prediction changes at other codes are small: about +1 mg/dL for lab glucose, and 0.1–0.25 mmHg for MAP. These come from the dim and unit ablations, not from the readout. `rollout_gen` returns a tuple (a list and a dict), and to calibrate I need to read it correctly. This run prints:
- the structure of that output on one stay;
- the keys of `mult`, the edit names `true_shift` uses;
- one comparison of model versus true shifts on that stay, if the format is obvious.

Next I'll run the calibration rollouts on about 50 tuning stays.

```python
L0,I0=Rg
print('list len', len(L0), 'elem', type(L0[0]).__name__, (getattr(L0[0],'shape',None) or (list(L0[0].keys())[:8] if isinstance(L0[0],dict) else len(L0[0]))))
print('dict keys', {k:(getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else v)) for k,v in list(I0.items())[:10]})
print('mult', mult if len(str(mult))<300 else list(mult.keys()))
```