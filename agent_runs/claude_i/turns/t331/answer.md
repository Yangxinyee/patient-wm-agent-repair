z_in is a plain linear layer (192×32), so a z dimension's effect on the prelude token is exactly one column. z_post is a 2-layer MLP (576→192→64), so its 50 dead hidden units can be repurposed. The plan is feasible.

/work/sim.py has a local whole-stay runner (`run_stay`) and the `zpost_mean` that hit the dtype error. If it reproduces the state update over a full stay, I can test candidate edits locally before spending forward passes. Next I'll read sim.py (it is short) and list what is stored in the saved arrays (probe_ds.npz, sIN.npy, Jtok.npz, G_tok_A.npy), so I can reuse earlier work instead of recomputing it.

```python
print(open('/work/sim.py').read())
for fn in ['probe_ds.npz','Jtok.npz']:
    d=np.load('/work/'+fn); print(fn, {k:d[k].shape for k in d.files})
for fn in ['sIN.npy','G_tok_A.npy','JI.npy','JN.npy']:
    print(fn, np.load('/work/'+fn).shape)
import pickle; F2=pickle.load(open('/work/F2.pkl','rb')); print(type(F2), (list(F2.keys())[:10] if isinstance(F2,dict) else len(F2)))
```