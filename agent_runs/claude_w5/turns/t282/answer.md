That is the cause. `WE` contains a large rank-2 edit to the value projection of prelude attention head 1 (rows 240–287, singular values 29k and 12k). This is the "stage 1" of my design, which amplifies the dose rate from the action token u into the slots. It was never saved as an edit.

The saved FFN decoders were presumably fitted on n3 with that stage-1 signal present. The tool runs without it, so they read noise and a state-driven offset. That gives the biased, dose-blind injection and the upward drift seen in fwd_4. (`run_cmp` did not fill a list, so that check yielded nothing.)

Before saving the kv tensor I want local confirmation, using `mystep` on a few training-stay insulin bundles, of how strongly the injected shift responds to doubling the dose under each set of weights:

| Weights | Expected response to the x2 dose (projected on vbar_I) |
|---|---|
| Original | none |
| `WE`, with the kv edit | about kappa × rate (≈ −4.6 × rate in reading units) |
| `WE` without the kv edit (what the tool ran) | weak |

This first run also prints the bundle and step structures, so that the code is robust if the layout differs from what I remember.

```python
import numpy as np, traceback
z2 = np.load('/work/fwd_2/states_original.npz')
sid_arr2 = z2['stay_id']; hasI2 = z2['has_insulin'].astype(bool)
sid0 = int(sid_arr2[np.where(hasI2)[0][0]])
B_ = bundles_B[sid0]; S_ = steps_all[sid0]
print('n bundles', len(B_), 'n steps', len(S_), 'bundle type', type(B_[0]).__name__)
print('bundle0', (list(B_[0].keys()) if isinstance(B_[0], dict) else str(B_[0])[:300]))
print('step keys', list(S_[0].keys()) if isinstance(S_[0], dict) else type(S_[0]))
idx_rows = np.where(sid_arr2==sid0)[0]
jI = [k for k,r in enumerate(idx_rows) if hasI2[r]]
print('insulin bundle idx', jI[:6])
print('insulin bundle:', str(B_[jI[0]])[:900])
print('WCUR keys same as WE?', (set(WCUR.keys())==set(WE.keys())) if isinstance(WCUR, dict) else type(WCUR))
print('WCUR kv == orig?', np.abs(to64(WCUR[kvn]) - to64(o_kv)).max() if isinstance(WCUR, dict) else None)
```