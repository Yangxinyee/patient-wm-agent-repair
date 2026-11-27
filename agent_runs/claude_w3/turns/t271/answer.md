The encoder layout matters: the dose atoms have their own encoder, giving `u_k`, which reaches the state only through prelude attention into E and then `b_in`. Next I'll list the signatures of my local helper functions so I can reuse the local prelude computation.

```python
import inspect
for fn in ['step','prelude_attn','pre_tokens','enc_bundle','stay_bundles','prev_state','prev_h','readout_truth','truth_rows','zpost_mean','zpost_err','check','check2','skeleton','build_bundles','bundle_vecs','ctx_of','atom_vec','set_enc','enc2','enc3','zp_enc2','zerr','zparts']:
    try:
        print(fn, str(inspect.signature(globals()[fn]))[:90])
    except Exception as ex:
        print(fn, 'ERR', type(ex).__name__)
```