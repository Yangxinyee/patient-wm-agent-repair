Core tensors: attention layers have no biases, every norm is a LayerNorm with weight and bias, and `b_in` and `c_out` are plain matrices. To size a dedicated memory channel I need the scale and stability of the residual stream (per-dim spread and per-bundle mean/std). I'll get these from the saved states.

```python
import numpy as np, glob
fs=sorted(glob.glob('/work/fwd_*/*'))
print(fs)
S=np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:S[k].shape for k in S.files})
```