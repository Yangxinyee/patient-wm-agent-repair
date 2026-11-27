The tensor list gives the full structure. The value head is a 192→192→17 MLP that reads `with_code`. The recurrent state passes through `c_out` and a one-layer coda transformer (attention plus FFN, with no bias on `c_out`) at every step.

I'll rebuild the local forward pass from scratch and check it against saved states. First I need the format of the saved state dumps and the call signatures of my old helpers, so I can reuse the data loaders.

```python
import inspect, glob, numpy as np
for d_ in sorted(glob.glob('/work/fwd_*')):
    print(d_, sorted(os.listdir(d_)))
z4 = np.load(glob.glob('/work/fwd_4/*.npz')[0], allow_pickle=True)
print({k: z4[k].shape for k in z4.files})
for fn in ['build_bundles','bundle_vecs','set_enc','atom_emb','mlp','LN','zpost_check2','zerr','A','G_','HN']:
    f_ = globals()[fn]
    try: print(fn, inspect.signature(f_), f_.__code__.co_names[:25])
    except Exception as e: print(fn, 'err', e)
```