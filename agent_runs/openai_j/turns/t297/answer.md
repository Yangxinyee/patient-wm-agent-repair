The cached head has 192 nonlinear units, offering a different feature space from the rejected scalar correction. Earlier contrast fits needed heavy regularization to preserve factual fit. I’ll measure the geometry around the validated edit before fitting anything. A distinct doubling direction with low factual cost would support a small statistical correction; collinearity would argue against it.

```python
import torch
import torch.nn.functional as F

weights259 = {
    name: torch.as_tensor(load(name), dtype=torch.float32).clone()
    for name in head177_names
}
fit259 = np.asarray(fit236, dtype=bool)
cv259 = ~fit259
events259 = np.asarray(events206)
print("Fit/CV readings:", int(fit259.sum()), int(cv259.sum()))
print("Fit events:", pd.Series(events259[fit259]).value_counts().to_dict())

datasets259 = {"train": wm_train145, "tune": wm_tune145}
datasets259.update(wm_cf146)
packs259 = {}
for key, mc in datasets259.items():
    cc = mc.get("centres", wm_train145["centres"])
    pred = head177_eval(mc, weights259, cc)
    hidden = F.gelu(F.linear(mc["inputs"],
                            weights259[head177_names[0]],
                            weights259[head177_names[1]]))
    packs259[key] = {"mc": mc, "pred": pred, "hidden": hidden}

hf259 = packs259["train"]["hidden"][:, fit259]
center259 = hf259.mean((0, 1))
scale259 = hf259.std((0, 1), unbiased=False).clamp_min(0.05)
v259 = torch.as_tensor(wm_v151, dtype=torch.float32)
for key, pack in packs259.items():
    hh = (pack["hidden"] - center259) / scale259
    zz = torch.cat([hh, torch.ones_like(hh[..., :1])], dim=-1)
    pp = pack["pred"]["probs"]
    cc = torch.as_tensor(
        pack["mc"].get("centres", wm_train145["centres"]),
        dtype=torch.float32)
    ev = (pp * v259).sum(-1)
    ec = (pp * cc).sum(-1)
    response = (pp * cc * v259).sum(-1) - ec * ev
    pack["Z"] = zz
    pack["response_features"] = (response[..., None] * zz).mean(0).numpy()
    pack["mean"] = (pp * cc).sum(-1).mean(0).numpy()

tr259 = packs259["train"]
p259 = tr259["pred"]["probs"]
z259 = tr259["Z"]
ev259 = (p259 * v259).sum(-1)
var259 = (p259 * v259.square()).sum(-1) - ev259.square()
target259 = torch.as_tensor(wm_train145["target"], dtype=torch.long)
pb259 = p259.gather(
    -1, target259[None, :, None].expand(p259.shape[0], -1, 1)
).squeeze(-1)
n259, d259 = z259.shape[1:]
assert p259.shape[0] == 16
check_nll259 = -pb259.reshape(4, 4, n259).mean(1).clamp_min(1e-12).log()
print("Mixture-NLL reconstruction max error:",
      float((check_nll259 - tr259["pred"]["nll"]).abs().max()))

num259 = (pb259[..., None] *
          (v259[target259][None, :] - ev259)[..., None] * z259)
gradrows259 = -(num259.reshape(4, 4, n259, d259).sum(1) /
                pb259.reshape(4, 4, n259).sum(1)[..., None])
grad259 = gradrows259[:, fit259].mean((0, 1)).numpy().astype(np.float64)
wh259 = (z259[:, fit259] *
         var259[:, fit259].clamp_min(0).sqrt()[..., None]).reshape(-1, d259)
H259 = (wh259.T @ wh259 / len(wh259)).numpy().astype(np.float64)

rows259, labels259, slopes259 = [], [], {}
sensors259 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
for drug in ["insulin", "norepinephrine"]:
    for factor in [0.5, 2.0]:
        key = (drug, factor)
        pack = packs259[key]
        active = np.asarray(pack["mc"]["active"], bool)
        dr = np.asarray(pack["mc"]["delta_rate"], np.float64)
        for sensor in sensors259:
            mask = fit259 & active & (events259 == sensor)
            denom = float(np.square(dr[mask]).sum())
            if denom <= 1e-12:
                continue
            D = pack["response_features"] - tr259["response_features"]
            rows259.append((D[mask] * dr[mask, None]).sum(0) / denom)
            label = (drug, factor, sensor)
            labels259.append(label)
            slopes259[label] = {}
            for label_subset, subset in [("fit", fit259), ("cv", cv259)]:
                mm = subset & active & (events259 == sensor)
                dd = float(np.square(dr[mm]).sum())
                slopes259[label][label_subset] = (
                    float(((pack["mean"] - tr259["mean"])[mm] * dr[mm]).sum()) / dd
                    if dd > 1e-12 else np.nan)
M259 = np.asarray(rows259, dtype=np.float64)

print("Moment design:", M259.shape)
for label in labels259:
    if label[0] == "norepinephrine" and label[2].startswith("map"):
        j = labels259.index(label)
        print(label, slopes259[label], "design norm", np.linalg.norm(M259[j]))

np.savez("/work/hidden_geometry259.npz",
         H=H259, grad=grad259, M=M259,
         center=center259.numpy(), scale=scale259.numpy())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 259: frozen-hidden statistical geometry\n"
            "- Computed factual score/curvature and paired dose-response moments in the current head's hidden features.\n"
            "- No gradients through the world model and no weights changed.\n"
            "- This is a feasibility screen, not intervention evidence.\n")
```