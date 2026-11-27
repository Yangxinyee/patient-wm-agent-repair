The pilot added only a small latent signal, while the original latent already changes substantially with dose. Simply amplifying it could amplify unrelated associations too. I’ll instead construct two explicit, gated dose-minus-policy signals and give them separate carrier directions. Held-out reconstruction accuracy and selective attention to that carrier will determine whether this is ready for a rollout test.

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import null_space
from pathlib import Path

n154 = len(idx81)
C154 = np.asarray(ctx81, dtype=np.float64)
E154 = np.asarray(e81, dtype=np.float64)
U154 = np.asarray(u81, dtype=np.float64)
Y154 = np.asarray(YY103[:n154], dtype=np.float64)
train154 = np.asarray(tr_bundle103, dtype=bool)
valid154 = ~train154
print("Bundles / held-out bundles:", n154, int(valid154.sum()))

policy154 = []
for drug154 in range(2):
    active154 = Y154[:, drug154+2] > 0.5
    tr154 = train154 & active154
    va154 = valid154 & active154
    cm154 = C154[tr154].mean(0)
    ym154 = Y154[tr154, drug154].mean()
    cc154 = C154[tr154] - cm154
    yy154 = Y154[tr154, drug154] - ym154
    cov154 = cc154.T @ cc154
    rhs154 = cc154.T @ yy154
    options154 = []
    for ridge154 in (0.01, 0.1, 1., 10., 100., 1000.):
        coef154 = np.linalg.solve(cov154 + ridge154*np.eye(192), rhs154)
        bias154 = float(ym154 - cm154 @ coef154)
        rmse154 = float(np.sqrt(np.mean(
            (C154[va154] @ coef154 + bias154 - Y154[va154, drug154])**2)))
        options154.append((rmse154, ridge154, coef154, bias154))
    best154 = min(options154, key=lambda v: v[0])
    policy154.append((best154[2], best154[3]))
    print("Policy", drug154, "active train/valid", int(tr154.sum()), int(va154.sum()),
          "ridge", best154[1], "held-out RMSE", best154[0])

post154 = {n: np.array(load_original(n), dtype=np.float32, copy=True)
           for n in post_original153}
rows154 = np.array([31, 74, 93, 105])
gate_strength154 = 40.
for drug154 in range(2):
    plus154, base154 = rows154[2*drug154:2*drug154+2]
    pc154, pb154 = policy154[drug154]
    rate154 = np.asarray(ld105[:, drug154], dtype=np.float64)
    presence154 = np.asarray(ld105[:, drug154+2], dtype=np.float64)
    for row154 in (plus154, base154):
        post154["z_post.0.weight"][row154] = 0
        post154["z_post.0.weight"][row154, 384:] = (
            2*gate_strength154*presence154[:192])
        post154["z_post.0.bias"][row154] = (
            2*gate_strength154*presence154[192] - gate_strength154)
    post154["z_post.0.weight"][plus154, :192] -= pc154.astype(np.float32)
    post154["z_post.0.weight"][plus154, 384:] += rate154[:192].astype(np.float32)
    post154["z_post.0.bias"][plus154] += np.float32(rate154[192] - pb154)

X154 = torch.cat((tensor153(C154), tensor153(E154), tensor153(U154)), -1)
H154 = F.gelu(F.linear(
    X154, tensor153(post154["z_post.0.weight"]),
    tensor153(post154["z_post.0.bias"]))).numpy().astype(np.float64)
Horig154 = F.gelu(F.linear(
    X154, post_original153["z_post.0.weight"],
    post_original153["z_post.0.bias"])).numpy().astype(np.float64)
Ptarget154 = Horig154 @ np.asarray(load_original("z_post.2.weight")).T + np.asarray(
    load_original("z_post.2.bias"))
baseW154 = np.asarray(load_original("z_post.2.weight"), dtype=np.float64).copy()
baseW154[:, rows154] = 0.
baseb154 = np.asarray(load_original("z_post.2.bias"), dtype=np.float64)
R154 = Ptarget154 - (H154 @ baseW154.T + baseb154)
hm154 = H154[train154].mean(0)
rm154 = R154[train154].mean(0)
hc154 = H154[train154] - hm154
rc154 = R154[train154] - rm154
dW154 = np.linalg.solve(hc154.T @ hc154 + 1e-4*np.eye(192), hc154.T @ rc154)
db154 = rm154 - hm154 @ dW154
post154["z_post.2.weight"] = (baseW154 + dW154.T).astype(np.float32)
post154["z_post.2.bias"] = (baseb154 + db154).astype(np.float32)
Pcomp154 = H154 @ post154["z_post.2.weight"].T + post154["z_post.2.bias"]
print("Held-out posterior compensation RMS:",
      float(np.sqrt(np.mean((Pcomp154[valid154]-Ptarget154[valid154])**2))))

zin_original154 = np.asarray(load_original("z_in.weight"), dtype=np.float64)
zb_original154 = np.asarray(load_original("z_in.bias"), dtype=np.float64)
mean_z154 = Ptarget154[train154, :32].mean(0)
sd_z154 = Ptarget154[train154, :32].std(0)
cost154 = sd_z154 * np.linalg.norm(zin_original154, axis=0)
reserved154 = np.argsort(cost154)[:2]
print("Reserved latent coordinates:", reserved154.tolist(),
      "SDs", sd_z154[reserved154], "removal costs", cost154[reserved154])

for drug154, coord154 in enumerate(reserved154):
    plus154, base154 = rows154[2*drug154:2*drug154+2]
    post154["z_post.2.weight"][coord154] = 0.
    post154["z_post.2.weight"][coord154, plus154] = 1.
    post154["z_post.2.weight"][coord154, base154] = -1.
    post154["z_post.2.bias"][coord154] = 0.

gamma154 = np.asarray(load_original("core.prelude.0.norm1.weight"), dtype=np.float64)
beta154 = np.asarray(load_original("core.prelude.0.norm1.bias"), dtype=np.float64)
assert np.min(np.abs(gamma154)) > 1e-5
constraints154 = np.c_[
    zin_original154, zb_original154, np.ones(192),
    beta154/gamma154, np.asarray(gate147)/gamma154
]
carrier154 = null_space(constraints154.T)[:, :2]
carrier_scale154 = 0.02
zin154 = zin_original154.copy()
zb154 = zb_original154 + zin_original154[:, reserved154] @ mean_z154[reserved154]
zin154[:, reserved154] = carrier_scale154 * carrier154

Z154 = posterior153(C154, E154, U154, {n:tensor153(v) for n,v in post154.items()}).numpy()
raw_latent154 = Z154 @ zin154.T + zb154
sigma154 = np.sqrt(raw_latent154.var(axis=1) + 1e-5)
sigma_ref154 = float(np.mean(sigma154[train154]))
normalized_latent154 = (
    (raw_latent154-raw_latent154.mean(1, keepdims=True))/sigma154[:,None]
    * gamma154 + beta154
)
value_extract154 = (carrier154/gamma154[:,None]) * sigma_ref154/carrier_scale154
payload154 = normalized_latent154 @ value_extract154

print("Carrier normalization relative SD:", float(sigma154.std()/sigma154.mean()))
print("Carrier extraction RMSE:",
      np.sqrt(np.mean((payload154-Z154[:,reserved154])**2, axis=0)))
print("Factual payload mean / SD:",
      payload154.mean(0), payload154.std(0))
print("Latent gate min/mean/max:",
      np.min(normalized_latent154 @ gate147),
      np.mean(normalized_latent154 @ gate147),
      np.max(normalized_latent154 @ gate147))
print("Gated signal vs expected innovation RMSE:")
for drug154 in range(2):
    pc154, pb154 = policy154[drug154]
    expected154 = (Y154[:,drug154] - C154 @ pc154 - pb154) * Y154[:,drug154+2]
    print(drug154, float(np.sqrt(np.mean(
        (Z154[:,reserved154[drug154]]-expected154)**2))))

print("Calibration helper references:",
      memory82.__code__.co_names, fn144.__code__.co_names)
print("Calibration arrays:")
for name, obj in list(globals().items()):
    if name.endswith("136") and (isinstance(obj, np.ndarray) or torch.is_tensor(obj)):
        print(name, tuple(obj.shape))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 154; forwards 10. Constructed, but did not install, "
        "four-neuron gated physical-dose innovations with fresh "
        "whole-stay-held-out context policies, compensated posterior "
        "outputs, and two explicit orthogonal latent carrier directions. "
        "Candidate dictionaries post154/zin154/zb154 retained. "
        "Transport and physiological output calibration still required; "
        "current five zero-payload edits unchanged.\n"
    )
```