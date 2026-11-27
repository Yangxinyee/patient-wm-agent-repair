I have encoder and head helpers but no full core-step reconstruction. The non-core tensors confirm that actions enter in only two ways: as a token in the prelude attention, and through the z_post MLP into the z_in token, which also goes through the prelude. So the prelude is the gateway for any dose signal.

Before deciding how much reconstruction is worth doing, I want to see where along that gateway the action information is still clean. I'll check the recorded states after action bundles (fwd_2), for the 32-dim posterior z versus out and the h slots. For each, I'll measure the cross-validated R² of the current action-A and action-B rates, raw and residualised on readings. If z is much cleaner than out, the information is lost in the prelude and slot update, and the edit should strengthen that path. If z is equally weak, the encoder itself discards it.

```python
S2 = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print({k: S2[k].shape for k in S2.files})
for k in S2.files:
    if S2[k].ndim == 1: print(k, S2[k][:4])
```