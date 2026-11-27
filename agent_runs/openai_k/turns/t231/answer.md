This repeats run 209, so there is no new validation result. The official history still shows that the routing is too weak or has the wrong sign. I’ll inspect the reserved-channel neurons next; a guarded correction is worth testing only if it separates edited doses from factual dose variation.

```python
import numpy as np
import torch
import dis

def a210(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("=== PHYSICAL-CHANNEL NEURONS ===")
p0 = a210(baseline138["z_post.0.weight"])
b0 = a210(baseline138["z_post.0.bias"])
p2 = a210(baseline138["z_post.2.weight"])
b2 = a210(baseline138["z_post.2.bias"])
for k in (30,31):
    active = np.flatnonzero(np.abs(p2[k]) > 1e-7)
    print("channel",k,"bias",float(b2[k]),"active",active.tolist(),
          "weights",p2[k,active].round(6).tolist())
    for j in active[:8]:
        print("hidden",int(j),"bias",float(b0[j]),"ctx/e/u norms",
              np.linalg.norm(p0[j].reshape(3,192),axis=1).round(5).tolist())

print("\n=== RETAINED DOSE ARRAYS ===")
for name in ("rr33_stays","rr33_hours","rr33_u_all","rr33_e_all",
             "r132_eligible","r132_drug","r132_mask","r132_ix",
             "r132_dz","r132_expected","r132_slopes","r132_error",
             "r132_rate","r132_factor","r132_z","_allz206","_hasaction206"):
    x = globals().get(name)
    if torch.is_tensor(x) or isinstance(x,np.ndarray):
        aa = a210(x)
        print(name,aa.shape,aa.dtype,"sample",repr(aa.reshape(-1)[:5])[:100])
    elif isinstance(x,dict):
        print(name,"keys",list(x)[:10])
    else:
        print(name,type(x).__name__)

print("\n=== REPLAY INTERFACES ===")
for name in ("latent205","r132_channels","local_token154","dynamic_delta154",
             "r186_heads","r186_means","r131_f","posterior_token154"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn,"__code__"):
        c = fn.__code__
        print(name,"args",c.co_varnames[:c.co_argcount],"uses",c.co_names[:30])

print("\n=== CHANNEL CALCULATION BYTECODE ===")
for name in ("latent205","r132_channels"):
    fn = globals().get(name)
    if callable(fn):
        text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn))
        print(name,"\n"+text[:2400])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 210: assess guarded physical-dose routing\n"
            "- Repeated run-209 output supplied no new validation. Official tested edits remain far from record-derived targets.\n"
            "- Inspecting reserved neurons and replay interfaces to assess stronger routing guarded against factual dose variation.\n"
            "- No weights changed. Python runs: 210; forward passes: 14.\n")
```