The input-level decoder recovers doubled norepinephrine and gives substantially better dose scaling across the other multipliers. Drug-presence gating is also accurate on held-out stays. I’ll install a reversible posterior branch, compensating for the neurons it replaces. The GPU test must show that its latent correction actually changes readings without exceeding the factual-loss limit.

```python
import numpy as np, torch, pandas as pd
from scipy.linalg import solve
from pathlib import Path

rp_old_w0=torch.as_tensor(load_original("z_post.0.weight")).numpy().copy()
rp_old_b0=torch.as_tensor(load_original("z_post.0.bias")).numpy().copy()
rp_old_w2=torch.as_tensor(load_original("z_post.2.weight")).numpy().copy()
rp_old_b2=torch.as_tensor(load_original("z_post.2.bias")).numpy().copy()
rp_fallback_numeric=torch.as_tensor(load("embed.numeric.weight")).clone()

_C=np.cov(rp_H[rp_train],rowvar=False,bias=True)+1e-4*np.eye(192)
_P=np.linalg.inv(_C)
_score=np.sum(rp_old_w2**2,axis=0)/np.diag(_P)
rp_reserved=np.argsort(_score)[:4]
rp_remaining=np.array([i for i in range(192) if i not in rp_reserved])
_Ar=np.column_stack([rp_H[:,rp_remaining],np.ones(len(rp_H))])
_ridge=1e-4*np.eye(_Ar.shape[1]);_ridge[-1,-1]=1e-7
rp_comp=solve(_Ar[rp_train].T@_Ar[rp_train]/rp_train.sum()+_ridge,
              _Ar[rp_train].T@rp_H[rp_train][:,rp_reserved]/rp_train.sum(),assume_a="pos")
rp_comp_w2=rp_old_w2.copy()
rp_comp_w2[:,rp_remaining]+=rp_old_w2[:,rp_reserved]@rp_comp[:-1].T
rp_comp_b2=rp_old_b2+rp_old_w2[:,rp_reserved]@rp_comp[-1]
rp_comp_w2[:,rp_reserved]=0
_original=rp_H[rp_hold]@rp_old_w2.T+rp_old_b2
_compensated=rp_H[rp_hold]@rp_comp_w2.T+rp_comp_b2
print("reserved posterior neurons:",rp_reserved.tolist())
print("held-out compensation RMSE means/log-SD:",
      np.sqrt(np.mean((_original[:,:32]-_compensated[:,:32])**2)),
      np.sqrt(np.mean((_original[:,32:]-_compensated[:,32:])**2)))

rp_branch_specs={}
for _j,(_drug,_event,_beta,_threshold) in enumerate((
    ("insulin","glucose_finger",-1.850,0.20),
    ("norepinephrine","map_arterial",3.351,-0.05))):
    _r,_rb=rp_linear_fits[(_drug,0.1)]
    _gate,_gb=rp_gate_fits[_drug]
    _gw=np.zeros(576);_gw[384:]=64*_gate
    _gb=64*(_gb-_threshold)
    rp_branch_specs[_drug]=dict(rows=rp_reserved[2*_j:2*_j+2],
        rw=_r,rb=_rb,gw=_gw,gb=_gb,
        physical=np.asarray(r168_physdirections[_event])*_beta)

def rp_install(gain_insulin,gain_norepinephrine):
    w0=rp_old_w0.copy(); b0=rp_old_b0.copy()
    w2=rp_comp_w2.copy(); b2=rp_comp_b2.copy()
    for drug,gain in (("insulin",gain_insulin),("norepinephrine",gain_norepinephrine)):
        s=rp_branch_specs[drug]; a,b=s["rows"]
        w0[a]=s["gw"]+s["rw"];b0[a]=s["gb"]+s["rb"]
        w0[b]=s["gw"];b0[b]=s["gb"]
        w2[:32,a]=gain*s["physical"];w2[:32,b]=-gain*s["physical"]
    drop_edit("embed.numeric.weight")
    for name,arr in (("z_post.0.weight",w0),("z_post.0.bias",b0),
                     ("z_post.2.weight",w2),("z_post.2.bias",b2)):
        save_edit(name,torch.tensor(arr,dtype=torch.float32))
    return w0,b0,w2,b2

rp_current_gains=(4.0,4.0)
rp_current_weights=rp_install(*rp_current_gains)
_rh=torch.nn.functional.gelu(torch.tensor(rp_X,dtype=torch.float32)@
                            torch.tensor(rp_current_weights[0]).T+
                            torch.tensor(rp_current_weights[1])).numpy()
_rz=_rh@rp_current_weights[2].T+rp_current_weights[3]
print("offline held-out posterior change RMS:",
      np.sqrt(np.mean((_rz[rp_hold,:32]-_original[:,:32])**2)))
print("CURRENT EDITS:",edited())
print("NEXT FORWARD STAYS:",[int(x) for x in ds_eval_ids])
_paths=list(Path("/work/fwd_9").glob("dose_edit_*.csv"))
if _paths:
    _df=pd.read_csv(_paths[0])
    print("previous rollout columns:",list(_df.columns))
    print(_df.head(2).to_string(index=False))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 176\nInstalled reversible gated dose-residual posterior branch, gains I=4 and NE=4. Four low-impact GELU neurons repurposed; original removed contributions compensated by a held-out-checked regression. Numeric 5x fallback preserved in rp_fallback_numeric and restored to original in the active candidate. Edits: z_post.0/.2 weights and biases. Awaiting GPU factual and rollout validation.\n")
```