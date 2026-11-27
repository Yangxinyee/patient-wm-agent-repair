The recent objects are intact, including the calibrated dose decoders and the edit installer. I’ll measure the correction directly on factual and perturbed action inputs. If it tracks dose changes but remains small, the latent pathway is weak; if it does not track them, the feature construction is faulty.

```python
import inspect
import torch.nn.functional as F
torch.set_num_threads(8)

print("=== INSTALLER ===")
print("signature:", inspect.signature(install_repair127))
try:
    print(inspect.getsource(install_repair127)[:3000])
except (OSError, TypeError):
    print("names:", install_repair127.__code__.co_names)
    print("constants:", repr(install_repair127.__code__.co_consts)[:1800])
print("policy scores:", policy_scores126)
print("slots:", repair_slots127)
for name in ("direction_fit104","direction_all104","dose_directions50",
             "repair_policyB127","policy_innov126","policy_mean126"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {str(k): (tuple(v.shape) if hasattr(v,"shape") else str(v)[:160])
                     for k,v in obj.items()})
    elif hasattr(obj,"shape"):
        print(name, tuple(obj.shape))
    else:
        print(name, type(obj).__name__)

def q130(x):
    return np.round(np.quantile(np.asarray(x), [.05,.5,.95]), 5).tolist()

Xbase130 = repair_input127.copy()
Xbase130[:,384:] = action_U122
print("ctx agreement:", float(np.sqrt(np.mean((Xbase130[:,:192]-policy_C126)**2))))
print("decoded rates range:", [q130(dose_hat126[:,j]) for j in range(2)])
print("gate predictions, by presence:")
G130 = action_U122 @ gate_W126
for j in range(2):
    for present in (False,True):
        mask = (action_Y122[:,j] > 0) == present
        print(j, present, int(mask.sum()), q130(G130[mask,j]))

W2130 = np.asarray(load("z_post.2.weight"))
b2130 = np.asarray(load("z_post.2.bias"))
def post_probe130(X):
    H0 = F.gelu(torch.from_numpy((X @ repair_W0127.T + repair_b0127).astype(np.float32))).numpy()
    H1 = F.gelu(torch.from_numpy((X @ repair_newW0127.T + repair_newb0127).astype(np.float32))).numpy()
    Y0 = H0 @ repair_W2127.T + repair_b2127
    Y1 = H1 @ W2130.T + b2130
    return Y1-Y0, H1[:,repair_slots127]

delta_base130, phi_base130 = post_probe130(Xbase130)
print("factual latent correction RMS:", np.sqrt(np.mean(delta_base130[:,:32]**2)))
print("factual latent norm quantiles:", q130(np.linalg.norm(delta_base130[:,:32],axis=1)))
for j in range(2):
    i,k = 2*j,2*j+1
    print("factual gated innovation", j,
          q130((phi_base130[:,i]-phi_base130[:,k])/repair_alpha127))
for drug,j in (("insulin",0),("norepinephrine",1)):
    mask = action_Y122[:,j] > 0
    for factor in (0.0,0.5,1.5,2.0,3.0):
        X = Xbase130.copy()
        X[:,384:] = action_edit_U123[(drug,factor)]
        dz,phi = post_probe130(X)
        dphi = (phi[:,2*j]-phi[:,2*j+1]-phi_base130[:,2*j]+phi_base130[:,2*j+1])/repair_alpha127
        true_dr = action_Y122[:,j]*(factor-1)
        slope = np.dot(dphi[mask],true_dr[mask])/np.dot(true_dr[mask],true_dr[mask])
        print(drug, factor, "feature slope", round(float(slope),5),
              "delta-z norm", q130(np.linalg.norm((dz-delta_base130)[mask,:32],axis=1)))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 129–130 — direct correction audit\n"
            "- Recovered candidate objects and inspecting actual posterior corrections under dose perturbations.\n"
            "- No new weights installed; first candidate remains active only for diagnosis.\n")
```
