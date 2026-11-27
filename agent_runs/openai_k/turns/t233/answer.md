There is a concrete replay discrepancy: empty action bundles currently have a nonzero embedding, whereas the architecture specifies zero. That could explain why local calibration disagreed with official rollouts. I’ll compare posterior means against official states with and without zeroing those bundles; a substantially closer match would confirm the mismatch.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)
s212 = np.load("/work/fwd_3/states_original.npz")
sid212 = np.asarray(rr33_stays)
hour212 = np.asarray(rr33_hours)
assert len(sid212) == len(s212["x"])
print("=== STATE ALIGNMENT ===")
print("stay match",np.array_equal(sid212,s212["stay_id"]),
      "hour max error",float(np.max(np.abs(hour212-s212["hour"]))))

ctx212 = torch.as_tensor(contexts_audit,dtype=torch.float32)
e212 = torch.as_tensor(rr33_e_all,dtype=torch.float32)
u212 = torch.as_tensor(rr33_u_all,dtype=torch.float32)
drug212 = s212["has_insulin"] | s212["has_norepinephrine"]
u_zero212 = u212.clone()
u_zero212[torch.as_tensor(~drug212)] = 0
truth212 = torch.as_tensor(s212["x"][:,-32:],dtype=torch.float32)

wo212 = {n:torch.as_tensor(load_original(n),dtype=torch.float32)
         for n in ("z_post.0.weight","z_post.0.bias",
                   "z_post.2.weight","z_post.2.bias")}

print("\n=== ORIGINAL POSTERIOR RECONSTRUCTION ===")
posterior_check212 = {}
for label,u in (("existing_empty_encoding",u212),
                ("zero_empty_action",u_zero212)):
    x = torch.cat((ctx212,e212,u),dim=1)
    hidden = F.gelu(F.linear(x,wo212["z_post.0.weight"],wo212["z_post.0.bias"]))
    mu = F.linear(hidden,wo212["z_post.2.weight"],wo212["z_post.2.bias"])[:,:32]
    err = (mu-truth212).numpy()
    posterior_check212[label] = mu
    print(label,"overall_RMS",float(np.sqrt(np.mean(err**2))),
          "no_drug_RMS",float(np.sqrt(np.mean(err[~drug212]**2))),
          "drug_RMS",float(np.sqrt(np.mean(err[drug212]**2))),
          "max_abs",float(np.abs(err).max()))

print("\n=== CURRENT RESERVED CHANNELS UNDER BOTH INPUT CONVENTIONS ===")
for label,u in (("existing_empty_encoding",u212),("zero_empty_action",u_zero212)):
    x = torch.cat((ctx212,e212,u),dim=1)
    hidden = F.gelu(F.linear(x,torch.as_tensor(baseline138["z_post.0.weight"]),
                             torch.as_tensor(baseline138["z_post.0.bias"])))
    mu = F.linear(hidden,torch.as_tensor(baseline138["z_post.2.weight"]),
                   torch.as_tensor(baseline138["z_post.2.bias"]))[:,:32]
    for group,mask in (("no_drug",~drug212),("drug",drug212)):
        a = mu.numpy()[mask,30:32]
        print(label,group,"n",len(a),"mean",a.mean(0).round(5),
              "std",a.std(0).round(5),
              "range",np.quantile(a,[0,.5,1],axis=0).round(5))

print("\n=== EARLY POSTERIOR/CORE VALIDATION NOTES ===")
note212 = Path("/work/notes.md").read_text()
lines212 = note212.splitlines()
early212 = [line for line in lines212[:650]
            if any(t in line.lower() for t in
                   ("posterior","empty","exact","core replay","core match"))]
for line in early212[:16]:
    print(line[:220])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 212: verify empty-action convention against official states\n"
            "- Local empty-action vectors have norm 11.67 despite architecture specifying zero; testing this as a replay mismatch, not assuming it is causal evidence.\n"
            "- Comparing original posterior means against official state dumps under both conventions.\n"
            "- No weights changed. Python runs: 212; forward passes: 14.\n")
```