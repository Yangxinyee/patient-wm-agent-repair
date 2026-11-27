Result: the original model's factual residuals at glucose readings still have clear slopes on the dose surprise. For finger glucose they are -1.5, -1.9 and -0.5 at lags 1 to 3, against true effects of about -2.0, -2.4 and -2.1. Lab glucose shows the same pattern, and the slope on the expected dose is about zero.

So the model leaves most of the lag-1 and lag-2 response out of its teacher-forced predictions. Its own lag-1 response (about -0.47 per unit) matches what the earlier rollout showed (-0.45). The plan is a centred head edit that supplies the missing response. It should leave factual NLL unchanged or improve it slightly.

Next I rebuild the offline value head from with_code and check it reproduces the stored predicted means and NLLs. If it matches to within the known z-sampling noise, I can design the edit offline.

```python
import torch
vb = [n for n in NAMES if 'value_bin' in n]
print(vb)
print({k: (np.shape(v) if hasattr(v,'shape') else type(v).__name__) for k,v in HI.items()})
```