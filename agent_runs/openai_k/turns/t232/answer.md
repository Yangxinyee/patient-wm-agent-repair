The visible output shows that each dose channel combines many posterior neurons, so shifting one bias would not create the proposed guard. I’ll identify the genuinely reserved neurons and check bundle flags against official states. This will distinguish a channel-design problem from an error in the local replay inputs.

```python
import numpy as np
import torch
import dis
from pathlib import Path

p0 = a210(baseline138["z_post.0.weight"])
p2 = a210(baseline138["z_post.2.weight"])
b0 = a210(baseline138["z_post.0.bias"])
row_norm211 = np.linalg.norm(p0,axis=1)
print("=== LARGEST POSTERIOR INPUT ROWS ===")
for j in np.argsort(row_norm211)[-10:][::-1]:
    print(int(j),"input_norm",round(float(row_norm211[j]),4),
          "bias",round(float(b0[j]),5),
          "other_mean_output_norm",round(float(np.linalg.norm(p2[:30,j])),6),
          "dose_outputs",p2[30:32,j].round(6).tolist())

print("\n=== OFFICIAL STATE FILE LAYOUT ===")
for folder in ("fwd_3","fwd_10"):
    p = Path("/work")/folder/"states_original.npz"
    if p.exists():
        z = np.load(p)
        print(folder,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
        for k in z.files:
            if z[k].ndim == 1:
                print(k,"sample",repr(z[k][:8])[:160])

print("\n=== DOSE DIAGNOSTIC ARRAYS ===")
for name in ("r132_eligible","r132_drug","r132_mask","r132_ix",
             "r132_dz","r132_expected","r132_slopes","r132_error",
             "r132_rate","r132_factor","_hasaction206"):
    x = globals().get(name)
    if torch.is_tensor(x) or isinstance(x,np.ndarray):
        aa = a210(x)
        print(name,aa.shape,str(aa.dtype),"sample",repr(aa.reshape(-1)[:5])[:110])
    elif isinstance(x,dict):
        print(name,"keys",list(x)[:8])
    else:
        print(name,type(x).__name__)

print("\n=== ACTION EMBEDDING COUNTS ===")
ua211 = a210(rr33_u_all)
norms211 = np.linalg.norm(ua211,axis=1)
print("zero vectors",int((norms211<1e-8).sum()),"total",len(norms211),
      "norm quantiles",np.quantile(norms211,[0,.25,.5,.75,1]).round(5))
if "empty_action_audit" in globals():
    print("empty_action norm",float(np.linalg.norm(a210(empty_action_audit))))

print("\n=== CHANNEL FUNCTION BYTECODE ===")
for name in ("latent205","r132_channels"):
    fn = globals().get(name)
    if callable(fn):
        text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn))
        print(name,"\n"+text[:2000])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 211: distinguish reserved neurons and bundle-input errors\n"
            "- Run 210 output truncated; visible dose channels are distributed posterior readouts, not simple two-neuron gates.\n"
            "- Checking reserved input rows and official bundle flags before altering their construction.\n"
            "- No weights changed. Python runs: 211; forward passes: 14.\n")
```