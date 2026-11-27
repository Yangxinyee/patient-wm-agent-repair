The replica functions are intact. `step`, `encode`, `bundle_vecs` and the atom builders reference live globals (`W`, `Wt`, `CFG`), so I can reuse them. `step` takes `z_k` as an input, though: the z posterior is computed outside it, and for counterfactual doses I must compute it myself from `[ctx, e_k, u_k]`.

Next I will check the stored config `CFG`, whether the earlier states arrays still exist, and what fwd_3's states file holds. Then I can run the replica on one stay, including drug bundles, and compare `out`/z with the tool. If the errors are tiny at drug bundles, I can generate counterfactual states. If not, I will move to a head-level approach on the tool's factual states.

```python
import numpy as np, json
print('CFG', CFG if not hasattr(CFG,'keys') else {k:CFG[k] for k in list(CFG)[:12]})
print([n for n in ['Xt','BUN','W','Wt','e_empty','u_empty','codes_','cen_'] if n in globals()], type(W).__name__, len(W))
S=np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:(S[k].shape, S[k].dtype) for k in S.files})
```