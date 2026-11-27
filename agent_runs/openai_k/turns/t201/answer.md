The head correction was fitted to the original model’s weak dose signal, which explains why amplification mostly added factual noise. I’ll now test the reserved latent channels directly: whether they encode a physical dose change, and whether that change survives the state update. Accurate latent contrasts would support rerouting them rather than enlarging the head edit.

```python
import numpy as np, torch, time
from pathlib import Path

print("=== RESERVED CHANNEL OBJECTS ===")
print("candidate keys:", list(candidates90))
print("zero-correction tensors:", list(candidates90[0.0]))
print("r132_channels:",
      {str(k): (tuple(v.shape) if hasattr(v, "shape") else v)
       for k, v in r132_channels.items()} if isinstance(r132_channels, dict)
      else (type(r132_channels).__name__, getattr(r132_channels, "shape", None)))

base185 = candidates90[0.0]
zin_delta185 = np.asarray(load("z_in.weight")) - np.asarray(base185["z_in.weight"])
cols185 = np.argsort(np.linalg.norm(zin_delta185, axis=0))[-4:]
print("changed z_in columns:", [(int(j), float(np.linalg.norm(zin_delta185[:, j]))) for j in cols185])
print("W0 currently original:",
      {k: bool(np.allclose(np.asarray(W0[k]), np.asarray(load_original(k))))
       for k in base185})
print("sizes:", len(r179_indices), tuple(r179_u0.shape))
print("mlp0 args:", mlp0.__code__.co_varnames[:mlp0.__code__.co_argcount])

backup185 = {k: W0[k].clone() for k in base185}
oldlin185 = lin0
captures185 = []
def capture185(x, prefix):
    y = oldlin185(x, prefix)
    if prefix == "z_in":
        captures185.append((x.detach().clone(), y.detach().clone()))
    return y

try:
    for k, v in base185.items():
        W0[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32).clone()
    uc185 = r179_u0.clone()
    for j185, drug185 in enumerate(["insulin", "norepinephrine"]):
        mask185 = r179_drug == j185
        uc185[torch.as_tensor(mask185)] = rr35_scaled_action_encoding(
            r179_indices[mask185], drug185, 2.0)
    globals()["lin0"] = capture185
    h0185, o0185 = r59_core(r179_indices, r179_u0)
    hc185, oc185 = r59_core(r179_indices, uc185)
finally:
    globals()["lin0"] = oldlin185
    for k, v in backup185.items():
        W0[k] = v

print("captured z/token shapes:", [(tuple(z.shape), tuple(t.shape)) for z,t in captures185])
if len(captures185) == 2:
    z0185, tok0185 = captures185[0]
    zc185, tokc185 = captures185[1]
    for j185, drug185 in enumerate(["insulin", "norepinephrine"]):
        m185 = r179_drug == j185
        dc185 = np.asarray(r179_dc[m185], dtype=float)
        dz185 = np.asarray(zc185-z0185)[m185]
        slopes185 = (dc185 @ dz185) / (dc185 @ dc185)
        print(drug185, "latent dose slopes:",
              [(int(k), float(slopes185[k])) for k in np.argsort(np.abs(slopes185))[-5:]],
              "reserved factual SD:", np.std(np.asarray(z0185)[m185][:, cols185], axis=0).tolist(),
              "token mean norm:", float(torch.linalg.vector_norm(tok0185[m185], dim=1).mean()))
print("replay return shapes:", tuple(h0185.shape), tuple(o0185.shape))
with open("/work/notes.md", "a") as f185:
    f185.write("\n## Run 185: reserved physical-dose channel assay\n"
               "- Original-head correction has little counterfactual signal; testing reserved posterior dose contrasts directly.\n"
               "- Scratch weights and monkeypatch restored after replay; saved edits unchanged.\n"
               "- Python runs: 185; forward passes: 10.\n")
```