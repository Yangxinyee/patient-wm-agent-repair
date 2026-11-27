Forward pass 10 explains the failure. Writer v2 does move the z posterior at insulin bundles, but only by about 0.03 rms per dimension. Downstream, the slot memory h changes by 0.03 against a natural spread of about 10.6, and out by 0.0005 against 0.26. The z token in the prelude is layer-normalised before attention, so the very large z_in columns cannot strengthen it, and the slots barely attend to it.

There is also no slowly decaying linear memory to use instead: a_bar lies between 0.29 and 0.43 in every dimension. Building a separate dose channel through the state is therefore a dead end, and I will drop the writer edit.

A better lever is the model's own dose pathway. The model already responds to insulin and norepinephrine in the right direction (insulin x2 −2.15, ½ +1.23), just too weakly. The dose reaches the state through the action encoder's output u_k. I can amplify the drug-rate-induced part of that encoder output, separately for each drug, with a linear (rank-limited) change to the encoder's last linear layer. A small change of direction in u stays in the linear regime, and the model's own downstream machinery then handles memory, propagation and readout. The gains k_insulin and k_norepinephrine can be calibrated with dose-edit passes. Factual cost is checked with factual passes.

First I need the encoder's internals and the helper functions already in the kernel. I also want the original model's per-hour response profile, to check that scaling the step effect gives a sensible accumulation.

```python
import inspect, numpy as np, pandas as pd
T = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
print([(n, tuple(SHAPES[n])) for n in NAMES if n.startswith(('encode_action', 'embed.'))])
for fn in ['datom', 'u_rec', 'u_c', 'setenc_my', 'view_from', 'gelu', 'ln']:
    f = globals().get(fn)
    if f is not None:
        try: print(fn, inspect.signature(f), f.__code__.co_varnames[:25])
        except Exception as ex: print(fn, ex)
print('vocab keys', list(VOCAB.keys()))
for key in VOCAB:
    v = VOCAB[key]
    if isinstance(v, dict) and ('insulin' in v):
        print(key, 'insulin:', str(v['insulin'])[:300]); print(key, 'ne:', str(v.get('norepinephrine'))[:300])
print('config', VOCAB.get('config'))
de = pd.read_csv('/work/fwd_11/dose_edit_original.csv'); print(de.columns.tolist()); print(de.head(2).to_string())
```