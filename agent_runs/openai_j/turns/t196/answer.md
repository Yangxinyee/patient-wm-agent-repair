The insulin correction uses two gated neurons, and the MAP candidate can use the same construction with the opposite sensor gate. I’ll allocate two additional neurons for a moderate MAP correction and check both factual loss and local responses. If those checks pass, I’ll test whether rollout amplification makes it useful.

```python
torch.set_num_threads(8)
head177_base = {k: v.clone() for k, v in best_insulin_v1.items()}
head177_names = [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
]
W0177, b0177, W2177, b2177 = [head177_base[k] for k in head177_names]

H177 = F.gelu(F.linear(wm_train145["inputs"], W0177, b0177))
h177 = H177.reshape(-1, 192).numpy().astype(np.float64)
A177 = np.column_stack([h177, np.ones(len(h177))])
G177 = A177.T @ A177 / len(A177)
P177 = np.linalg.inv(G177 + np.eye(193) * 1e-8)
wc177 = W2177.numpy() - W2177.numpy().mean(axis=0, keepdims=True)
score177 = (wc177**2).sum(axis=0) / np.maximum(np.diag(P177)[:192], 1e-12)
score177[np.asarray(trial158_free, dtype=int)] = np.inf
head177_free = np.argsort(score177)[:2]
head177_keep = np.setdiff1d(np.arange(192), head177_free)
K177 = A177[:, np.r_[head177_keep, 192]]
Y177 = h177[:, head177_free] @ W2177[:, head177_free].numpy().T
cov177 = K177.T @ K177 / len(K177)
comp177 = np.linalg.solve(
    cov177 + np.eye(len(head177_keep) + 1) * 1e-8,
    K177.T @ Y177 / len(K177)
)

head177_compressed = {k: v.clone() for k, v in head177_base.items()}
head177_compressed[head177_names[2]][:, head177_keep] += torch.tensor(
    comp177[:-1].T, dtype=torch.float32)
head177_compressed[head177_names[3]] += torch.tensor(comp177[-1], dtype=torch.float32)
head177_compressed[head177_names[2]][:, head177_free] = 0

def head177_make(map_coef):
    ww = {k: v.clone() for k, v in head177_compressed.items()}
    gate = -torch.as_tensor(trial158_gate, dtype=torch.float32)
    coef = torch.as_tensor(map_coef, dtype=torch.float32)
    j0, j1 = map(int, head177_free)
    ww[head177_names[0]][j0] = coef[:192] + 200 * gate[:192]
    ww[head177_names[1]][j0] = coef[-1] + 200 * gate[-1]
    ww[head177_names[0]][j1] = 200 * gate[:192]
    ww[head177_names[1]][j1] = 200 * gate[-1]
    ww[head177_names[2]][:, j0] = wm_v151
    ww[head177_names[2]][:, j1] = -wm_v151
    return ww

def head177_eval(mc, weights, centres=None):
    xx = mc["inputs"]
    hn = F.gelu(F.linear(xx, weights[head177_names[0]], weights[head177_names[1]]))
    ho = F.gelu(F.linear(xx, wm_weights145[head177_names[0]],
                        wm_weights145[head177_names[1]]))
    added = (
        F.linear(hn, weights[head177_names[2]], weights[head177_names[3]])
        - F.linear(ho, wm_weights145[head177_names[2]], wm_weights145[head177_names[3]])
    )
    logits = mc["logits"] + added
    probs = logits.softmax(-1)
    result = {"logits": logits, "probs": probs}
    if "target" in mc:
        target = mc["target"].long()
        pobs = probs.gather(-1, target[None, :, None].expand(probs.shape[0], -1, 1)).squeeze(-1)
        result["nll"] = -pobs.reshape(-1, 4, len(target)).mean(1).clamp_min(1e-30).log().mean(0).numpy()
    if centres is None:
        centres = mc.get("centres", wm_train145["centres"])
    result["mean"] = (probs.mean(0) * torch.as_tensor(centres)).sum(-1).numpy()
    return result

head177_coef = 1.5 * map172_candidates[(1.0, 1.0)]
head177_weights = head177_make(head177_coef)
print("Additional MAP neurons:", head177_free.tolist())
for label, mc in [("train", wm_train145), ("tuning", wm_tune145)]:
    zero = head177_eval(mc, head177_make(np.zeros(193)))
    old = head177_eval(mc, head177_base)
    new = head177_eval(mc, head177_weights)
    original = head177_eval(mc, wm_weights145)
    print(label, "compression_logit_RMS",
          float((zero["logits"] - old["logits"]).square().mean().sqrt()),
          "value_dNLL", float(np.mean(new["nll"] - original["nll"])),
          "max_mean_change_from_insulin_fallback",
          float(np.max(np.abs(new["mean"] - old["mean"]))))

head177_base_mu = head177_eval(wm_train145, head177_weights)["mean"]
for (drug, factor), mc in wm_cf146.items():
    out = head177_eval(mc, head177_weights)["mean"]
    target_sensor = "glucose_finger" if drug == "insulin" else "map_arterial"
    active = np.asarray(mc["active"], dtype=bool)
    select = active & (wm_events151 == target_sensor)
    dr = np.asarray(mc["delta_rate"])[select]
    dy = (out - head177_base_mu)[select]
    print(drug, factor, "local_shift", float(dy.mean()),
          "local_slope", float(dr @ dy / (dr @ dr)))

for name, tensor in head177_weights.items():
    save_edit(name, tensor)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 177: modest MAP extension installed for validation\n"
            f"- Preserved insulin fallback and allocated MAP neurons {head177_free.tolist()} by algebraic head compression.\n"
            "- Added shared MAP-gated raw-input tilt: fw=1, gain=1.5 from the stochastic factual/NE-contrast ridge fit.\n"
            "- All non-value-head weights remain original. GPU factual and dose-rollout checks are required next.\n")
print("Active edits:", edited())
```
