The output arrived but my reply was cut off, so I'll keep this reply short and technical.

ARCHITECTURE.md confirms how a drug bundle reaches the forecast. In the rollout, its posterior mean z feeds the state update through the token z_in(z) in the prelude, alongside the tokens e_k and u_k. The forecast heads then read the state, with z drawn from the prior.

So v1's z_post change should have reached later predictions. It produced essentially no shift. Either the state barely uses the z token, or my gates did not fire on the GPU's real inputs.

I can test the first explanation for free. In fwd_4, the probe set z_in.weight = 0, which removed the z route entirely. If those states nearly match the original states (fwd_3) on the same bundles, the z route has little leverage and I must inject elsewhere: the u_k token, the prelude, or the recurrent update. If they differ a lot, z matters and the gating is the suspect.

```python
import numpy as np
A = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
B = np.load('/work/fwd_4/states_edited.npz', allow_pickle=True)
print({k: A[k].shape for k in A.files})
print({k: B[k].shape for k in B.files})
```