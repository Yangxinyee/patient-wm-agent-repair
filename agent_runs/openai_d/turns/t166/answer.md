Active insulin states do cluster tightly around zero dose innovation, which makes thresholding worth investigating. However, the older cached formulas did not reproduce the installed head, so their loss estimates are unsafe to use. I’ll rebuild the comparison directly from the saved tensors. Agreement with the verified current loss will confirm the reconstruction before any new candidate is tested.

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax

r145_names = [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
]
def r145_t(x):
    return torch.as_tensor(r144_np(x), dtype=torch.float32)

r145_X = torch.as_tensor(r137_Xnp, dtype=torch.float32)
r145_orig = {n: load_original(n).clone() for n in r145_names}

def r145_hidden(w):
    return F.gelu(F.linear(
        r145_X, r145_t(w[r145_names[0]]), r145_t(w[r145_names[1]])
    ))

def r145_head(w, h=None):
    if h is None:
        h = r145_hidden(w)
    return F.linear(h, r145_t(w[r145_names[2]]), r145_t(w[r145_names[3]]))

r145_Hcurrent = r145_hidden(r139_candidate)
r145_transition = r145_t(r139_actual_L) - r145_head(r139_candidate, r145_Hcurrent)
r145_Horiginal = r145_hidden(r145_orig)
r145_Hins2 = r145_hidden(r115_strength2)

r145_Loriginal = r145_transition + r145_head(r145_orig, r145_Horiginal)
r145_Lins2 = r145_transition + r145_head(r115_strength2, r145_Hins2)
r145_Lcurrent = r145_transition + r145_head(r139_candidate, r145_Hcurrent)

r145_loss_orig = r144_nll(r145_Loriginal.numpy())
r145_loss_ins2 = r144_nll(r145_Lins2.numpy())
r145_loss_current = r144_nll(r145_Lcurrent.numpy())

print("=== EXACT LOCAL RECONSTRUCTION ===")
print("current vs verified loss max error:",
      float(np.max(np.abs(r145_loss_current-r139_loss))))
print("original vs verified loss max error:",
      float(np.max(np.abs(r145_loss_orig-r138_original_loss))))
print("strength-2 logits vs verified base max error:",
      float(torch.max(torch.abs(r145_Lins2-r145_t(r138_L0)))))

for name, loss in [("insulin strength 2", r145_loss_ins2),
                   ("combined trial", r145_loss_current)]:
    delta = loss-r145_loss_orig
    print(name, {"all": float(delta.mean()),
                 "fit": float(delta[r137_fit].mean()),
                 "held_out": float(delta[r137_val].mean())})

r145_W2orig = r145_orig[r145_names[2]]
r145_W2ins = r145_t(r115_strength2[r145_names[2]])
r145_old_ins = (
    r145_Horiginal[..., 107, None] * r145_W2orig[:, 107]
    + r145_Horiginal[..., 109, None] * r145_W2orig[:, 109]
)
r145_ins_bias = (
    r145_t(r115_strength2[r145_names[3]]) - r145_orig[r145_names[3]]
)
r145_Lins_removed = r145_Loriginal-r145_old_ins+r145_ins_bias
r145_Cins = (0.4*r145_W2ins[:, 107]).numpy()
print("insulin output pair cancellation:",
      float(torch.max(torch.abs(r145_W2ins[:,107]+r145_W2ins[:,109]))))

r145_ins_pair = (
    F.gelu(torch.as_tensor(r144_G+0.2*r144_D))
    - F.gelu(torch.as_tensor(r144_G-0.2*r144_D))
).numpy()/0.4
r145_pair_formula = (
    r145_Lins_removed.numpy()
    + r145_ins_pair[...,None]*r145_Cins
)
print("new insulin formula max logit error:",
      float(np.max(np.abs(r145_pair_formula-r145_Lins2.numpy()))))

r145_NEdelta = (r145_Lcurrent-r145_Lins2).numpy()
r145_removed_loss = r144_nll(r145_Lins_removed.numpy())
print("insulin neuron removal-only NLL delta:",
      float((r145_removed_loss-r145_loss_orig).mean()))
print("active insulin D/G correlation:",
      float(np.corrcoef(r144_D[:,r144_active].reshape(-1),
                        r144_G[:,r144_active].reshape(-1))[0,1]))
```