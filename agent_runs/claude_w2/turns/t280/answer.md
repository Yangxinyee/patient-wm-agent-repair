The previous output was refused by the content filter, so from now on I'll keep outputs short and print only summaries.

What run 266 established:
- **Insulin, true effect:** the per-lag effect on glucose is -2.0, -2.5, -2.2, -1.8, -1.4, -1.15, ... per U/h, summing to about -13.7. It is linear and the same for finger and lab glucose.
- **Norepinephrine, true effect:** the per-lag effect on MAP is +3.6, +2.7, +1.7, +1.1, +0.4, ..., summing to about +10.4. It is the same for arterial and cuff MAP.
- **Latest work:** the newest files (chan.py, corechan.npz, transfer_T.npy) point to a "dose memory channel" design that runs through the encoder and core.

Next I'll check how far that design got, using compact summaries:
- the contents of corechan.npz and the shapes of the saved .npy files;
- the comment lines in chan.py;
- the result variables still alive in the kernel.

If a finished design and its validation numbers exist, I'll continue from them. Otherwise I'll fall back to a simpler edit that only touches the value head.

```python
import numpy as np, types
z = np.load('/work/corechan.npz', allow_pickle=True)
print("corechan:", {k: z[k].shape for k in z.files})
for f in ['wI_chan.npy','wI_head2.npy','transfer_T.npy']:
    a = np.load('/work/'+f, allow_pickle=True); print(f, a.shape, a.dtype)
src = open('/work/chan.py').read().splitlines()
com = [l.strip()[:90] for l in src if l.strip().startswith('#') or l.startswith('def ')]
print("chan.py comments/defs:", len(com)); print("\n".join(com[:30]))
skip = (types.ModuleType, types.FunctionType, type)
items = []
for k, v in list(globals().items()):
    if k.startswith('_') or isinstance(v, skip) or k in ('In','Out','exit','quit','get_ipython'): continue
    if hasattr(v, 'shape'): d = f"{type(v).__name__}{tuple(v.shape)}"
    elif isinstance(v, (dict, list, tuple)): d = f"{type(v).__name__}[{len(v)}]"
    elif isinstance(v, (int, float, str)): d = repr(v)[:30]
    else: d = type(v).__name__
    items.append(f"{k}:{d}")
print(len(items), "vars:", "; ".join(items[-150:])[:3500])
```