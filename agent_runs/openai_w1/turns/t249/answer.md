Attention-output changes do reach the predicted state, but the measured gain is very small. That helps explain why the earlier latent edits had little effect. I’ll test value-head sensitivities and solve for two output directions that affect the intended measurement channels. Finite-amplitude tests must reproduce the estimated responses; otherwise, the local approximation is not usable.

```python
import numpy as np
import torch

print("=== STATE SCALES ===")
for nn221, tt221 in [
    ("input slots", rx_tokens126[:, :4]),
    ("flowed h", rx_hflow),
    ("prelude E", base220[0]),
    ("updated h", hbase220),
    ("out", obase220)
]:
    print(nn221, "RMS", float(torch.sqrt(torch.mean(tt221 ** 2))),
          "max", float(torch.max(torch.abs(tt221))))

def value_raw221(x):
    return rx_linear(torch.relu(rx_linear(x, "attr_heads.value_bin.0")),
                     "attr_heads.value_bin.2")

wc221 = torch.as_tensor(ce_head["withcode"], dtype=torch.float32)
phi221 = torch.relu(rx_linear(wc221, "attr_heads.value_bin.0"))
print("value hidden reconstruction RMSE:",
      float(torch.sqrt(torch.mean(
          (phi221 - torch.as_tensor(ce_head["phi"])) ** 2))))

events221 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
headidx221 = {}
qctx221 = {}
wcs221 = {}
logits221 = {}
centres221 = {}
rawbase221 = {}
G221 = np.empty((4, 48, 192), dtype=np.float32)
jv221 = torch.from_numpy(jac_v220)
gap221 = rx_mlp(torch.full((48, 1), np.log(60.0) / 10,
                           dtype=torch.float32), "gap", "silu")

for ee221, event221 in enumerate(events221):
    ix221 = np.flatnonzero(ce_head["rows"]["event"].to_numpy() == event221)
    prevout221 = np.asarray(ce_head["previous"])[ix221, :192]
    targetout221 = rx_targetout.numpy()
    distances221 = (
        np.sum(targetout221 ** 2, axis=1)[:, None]
        + np.sum(prevout221 ** 2, axis=1)[None, :]
        - 2 * targetout221 @ prevout221.T
    )
    chosen221 = ix221[np.argmin(distances221, axis=1)]
    headidx221[event221] = chosen221
    prior221 = torch.as_tensor(ce_head["prior"][chosen221, :32])
    qctx221[event221] = (
        vbase220 + gap221 + rx_linear(prior221, "z_head")
        + rx_W["type_emb.weight"][0]
    )
    wcs221[event221] = wc221[chosen221]
    logits221[event221] = torch.as_tensor(ce_head["logits"][chosen221])
    centres221[event221] = torch.as_tensor(ce_head["centres"][chosen221])
    rawbase221[event221] = value_raw221(wcs221[event221])
    for jj221 in range(192):
        dv221 = jv221[:, :, jj221] * eps220
        dq221 = (
            rx_norm(qctx221[event221] + dv221, "query_norm")
            - rx_norm(qctx221[event221] - dv221, "query_norm")
        ) / 2
        lp221 = logits221[event221] + (
            value_raw221(wcs221[event221] + dq221) - rawbase221[event221]
        )
        lm221 = logits221[event221] + (
            value_raw221(wcs221[event221] - dq221) - rawbase221[event221]
        )
        mup221 = (torch.softmax(lp221, -1) * centres221[event221]).sum(-1)
        mum221 = (torch.softmax(lm221, -1) * centres221[event221]).sum(-1)
        G221[ee221, :, jj221] = ((mup221 - mum221) / (2 * eps220)).numpy()

print("\n=== VALUE SENSITIVITY ===")
print("mean sensitivity row norms:", np.linalg.norm(G221.mean(axis=1), axis=1))
targets221 = np.array([[1., 0.], [1., 0.],
                       [0., 1.], [0., 1.]], dtype=np.float64)
M221 = G221[:, :36, :].reshape(-1, 192).astype(np.float64)
T221 = np.repeat(targets221[:, None, :], 36, axis=1).reshape(-1, 2)
penalty221 = 0.1 * np.trace(M221.T @ M221) / 192
B221 = np.linalg.solve(M221.T @ M221 + penalty221 * np.eye(192),
                      M221.T @ T221)
print("direction norms:", np.linalg.norm(B221, axis=0))
print("held-out local responses:",
      np.einsum("esd,dk->esk", G221[:, 36:], B221).mean(axis=1))

def finite_readout221(extra):
    rr221 = transition214(extra=np.asarray(extra, dtype=np.float32), act="gelu")
    vv221 = rr221[2] + rx_norm(rr221[1].mean(dim=1), "core.out_norm")
    change221 = vv221 - vbase220
    result221 = []
    for event221 in events221:
        dq221 = (
            rx_norm(qctx221[event221] + change221, "query_norm")
            - rx_norm(qctx221[event221], "query_norm")
        )
        ll221 = logits221[event221] + (
            value_raw221(wcs221[event221] + dq221) - rawbase221[event221]
        )
        mu221 = (torch.softmax(ll221, -1) * centres221[event221]).sum(-1)
        old221 = (torch.softmax(logits221[event221], -1)
                  * centres221[event221]).sum(-1)
        result221.append(float((mu221 - old221).mean()))
    return result221

print("\n=== FINITE-AMPLITUDE CHECKS ===")
for kk221 in range(2):
    for aa221 in [-1., 0.25, 1.]:
        print("channel", kk221, "amplitude", aa221,
              finite_readout221(B221[:, kk221] * aa221))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 221\n"
            "Attention-to-output Jacobian is small. Fitted two closed-form "
            "finite-difference response directions against glucose and MAP "
            "heads, with alternate-channel and held-out local checks. Testing "
            "finite amplitudes before installation. No weights changed. "
            "Counts: 221 Python runs, 20 forwards.\n")
```