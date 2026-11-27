The value, rate and other heads are small MLPs on the query. `pending_mlp` is used only for action prediction, so the observation heads get no dose channel outside the recurrent state. The door is the prelude attention from the slot tokens to the u_k (or z) token. Next I will check the saved state file fwd_4 (all bundles of 12 stays) and my local encoder, and measure how accurately I can rebuild the z-posterior for bundles with and without drug atoms. If the error is small for both kinds, my local u_k is trustworthy and I can compute the prelude's attention to it.

```python
S4=np.load('/work/fwd_4/states_original.npz', allow_pickle=True); print({k:S4[k].shape for k in S4.files})
import inspect
for fn in ['enc_bundle','stay_bundles','zpost_err']:
    f=globals().get(fn); print(fn, None if f is None else inspect.signature(f))
```