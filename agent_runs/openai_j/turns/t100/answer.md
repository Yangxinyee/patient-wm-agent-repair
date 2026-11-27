The rejected edit’s matrices are still cached, so I can audit them without changing the model. I’ll check feature centering and evaluate the correction under sampled priors. If a mean-state fit concealed large stochastic tilts, this should reproduce the factual loss increase.

```python
import numpy as np
import torch
import torch.nn.functional as TF

print("FEATURE COMPARISONS")
for name in ("repair_fH", "repair_base_H", "head_hidden"):
    arr = np.asarray(globals()[name])
    print(name, "RMS difference from repair_fH:",
          np.sqrt(np.mean((arr-repair_fH)**2)))

axis87 = np.asarray(repair_axis, dtype=np.float64)
decoder87 = np.asarray(repair_head_dW).T @ axis87 / (axis87 @ axis87)
intercept87 = float(np.asarray(repair_head_db) @ axis87 / (axis87 @ axis87))
print("Axis:", axis87)
print("Raw decoder norm / intercept:", np.linalg.norm(decoder87), intercept87)
print("Feature SD quantiles:", np.quantile(repair_head_sd, [0,.1,.5,.9,1]))

tau87_raw = repair_fH @ decoder87 + intercept87
tau87_fit = ((repair_fH-repair_head_mean)/repair_head_sd
             @ repair_head_coef + repair_head_intercept)
print("Raw vs fitted tilt RMS difference:",
      np.sqrt(np.mean((tau87_raw-tau87_fit)**2)))
print("Deterministic tilt RMS:", np.sqrt(np.mean(tau87_raw**2)))

head87 = reconstruct_reading_head(head_meta)
prior87 = mlp(head87["context"], "z_prior", "gelu")
sd87 = torch.exp(prior87[:,32:].clamp(-10, 5))
codes87 = head87["code"]
bins87 = head87["last_bin"]
transition87 = wt("embed_transition.value_bin.weight")[codes87*17+bins87]
dW87 = torch.as_tensor(repair_head_dW, dtype=torch.float32)
db87 = torch.as_tensor(repair_head_db, dtype=torch.float32)
target87 = torch.as_tensor(head_meta.target_bin.to_numpy(), dtype=torch.long)
rows87 = torch.arange(len(head_meta))
p87_original = torch.zeros((len(head_meta),17))
p87_candidate = torch.zeros_like(p87_original)
tilt_rms87 = []
torch.manual_seed(8701)

for draw87 in range(16):
    noise87 = torch.randn_like(sd87)*sd87
    view87 = head87["view"] + TF.linear(noise87, wt("z_head.weight"))
    input87 = (ln(view87 + wt("type_emb.weight")[1], "query_norm")
               + head87["register"])
    hidden87 = TF.gelu(TF.linear(input87,
                               wt("attr_heads.value_bin.0.weight"),
                               wt("attr_heads.value_bin.0.bias")))
    logits87 = TF.linear(hidden87, wt("attr_heads.value_bin.2.weight"),
                         wt("attr_heads.value_bin.2.bias")) + transition87
    change87 = TF.linear(hidden87, dW87, db87)
    p87_original += torch.softmax(logits87, -1)/16
    p87_candidate += torch.softmax(logits87+change87, -1)/16
    tau87 = change87 @ torch.as_tensor(axis87, dtype=torch.float32) / float(axis87 @ axis87)
    tilt_rms87.append(float(torch.sqrt(torch.mean(tau87**2))))

nll87_original = -torch.log(p87_original[rows87,target87].clamp_min(1e-30))
nll87_candidate = -torch.log(p87_candidate[rows87,target87].clamp_min(1e-30))
print("Sampled tilt RMS: min/mean/max",
      min(tilt_rms87), np.mean(tilt_rms87), max(tilt_rms87))
print("MC original/candidate NLL:",
      float(nll87_original.mean()), float(nll87_candidate.mean()))
print("MC candidate dNLL:", float((nll87_candidate-nll87_original).mean()))
print("ACTIVE EDITS:", edited())
```
