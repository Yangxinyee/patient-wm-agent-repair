The inventory shows the pieces I need are already built:
- z_post is a one-hidden-layer MLP (`z_post.0` → GELU → `z_post.2`) and z_in is linear.
- Per-bundle context, e and u arrays already exist: `C2`, `Eall` and `Uall` (5156 bundles).
- There are stored states (`X3`) and several helpers from earlier work.

The kernel helpers' source code can't be read, so I need their signatures before I can rely on them. I also need to confirm three things on the 5156 stored bundles:
- my numpy z_post reproduces the tool's z posterior mean (last 32 numbers of each state);
- exactly which hidden units are dead;
- how far the dead units' pre-activations sit below zero.

If the reproduction error is about 0 and roughly 50 dead units show up, the hijack plan is feasible. A large error would mean `C2`, `Eall` or `Uall` is stale and must be rebuilt from enc2.py.

```python
import inspect
for fn in ['core_step','step_u','step_local','run_stay','run_stay_z','zpost_cand','zpost_check','dev_ins','tgt_dev','pibar','best_dir','jac_at','value_mean','pred_mean_s','view_ctx','zprior','eu2','zpost2','ctx_of2','true_shift','policy_tables','hourly_rate','gelu_']:
    try: print(fn, inspect.signature(globals()[fn]))
    except Exception as ex: print(fn, 'ERR', str(ex)[:60])
zf = np.load('/work/fwd_3/states_original.npz'); print({k: zf[k].shape for k in zf.files})
W0=load('z_post.0.weight'); b0=load('z_post.0.bias'); W2=load('z_post.2.weight'); b2=load('z_post.2.bias'); Wzi=load('z_in.weight'); bzi=load('z_in.bias')
print(W0.shape, W2.shape, Wzi.shape, bzi.shape)
from scipy.special import erf
gel = lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))
Xin = np.concatenate([C2,Eall,Uall],1).astype(np.float64)
pre = Xin@W0.T.astype(np.float64) + b0
hid = gel(pre); zm = hid@W2.T + b2
print('z err', np.abs(zm[:,:32]-X3[:,960:]).max(), np.median(np.abs(zm[:,:32]-X3[:,960:])))
sd = hid.std(0); dead = np.where(sd<0.01)[0]; print('ndead', len(dead), 'pre max over dead', pre[:,dead].max().round(2), 'mean', pre[:,dead].mean().round(2))
print('z18/z28 sd', zm[:,18].std().round(4), zm[:,28].std().round(4), 'mean', zm[:,18].mean().round(3), zm[:,28].mean().round(3))
```