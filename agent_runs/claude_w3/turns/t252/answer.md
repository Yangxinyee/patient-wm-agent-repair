Disconnecting dimensions 49 and 185 costs +0.0034 NLL on 100 stays. A pair from the lower-influence set (9, 15, 149, 137) should cost roughly a quarter less. Those scores run 0.64–0.72, against 0.82 and 0.96 for the pair I tested. That still leaves too little margin unless the injection itself is cheap.

The larger problem is the way in. The infusion encoding u reaches the state only through prelude attention, so any injected signal gets scaled by each slot's attention weight on the u token. Before building anything, I'll recover the signatures of the helpers still in the kernel, so I can reuse the local encoder and prelude-attention code to measure:
- how well the code-8 and code-9 rates can be read linearly from the normalised u token;
- how stable the slot-to-u attention weights are.

```python
import inspect
for fn in ['stay_bundles','enc_bundle','prelude_attn','step','mha','pre_tokens','pre_probs','comp','ctx_of','enc3','zpost_mean','atom_vec','set_enc_v','lnorm']:
    try: print(fn, str(inspect.signature(globals()[fn]))[:120])
    except Exception as ex: print(fn,'ERR',type(ex).__name__)
```