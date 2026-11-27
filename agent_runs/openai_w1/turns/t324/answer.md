The original posterior reconstruction is also inaccurate, while its incoming-state reconstruction matches exactly. That narrows the error to the context or posterior calculation. I’ll compare cached contexts and activations against exported posterior means, including the distinction between context normalization and core output normalization. Near-numerical agreement is needed before continuing.

```python
def post284(context, action, weights, activation):
    xx = torch.cat((context, e277, action), dim=-1)
    hh = F.linear(xx, weights["z_post.0.weight"],
                  weights.get("z_post.0.bias"))
    if activation == "gelu":
        hh = F.gelu(hh)
    elif activation == "silu":
        hh = F.silu(hh)
    elif activation == "relu":
        hh = F.relu(hh)
    elif activation == "tanh":
        hh = torch.tanh(hh)
    else:
        raise ValueError(activation)
    return F.linear(hh, weights["z_post.2.weight"],
                    weights.get("z_post.2.bias"))[:, :32]

context_names284 = [
    "_ctx", "r168_ctx", "rp_ctx", "rr_best_ctx111", "rr_ctx4",
    "ctx277"
]
scores284 = []
for cname in context_names284:
    cc = globals().get(cname)
    if not isinstance(cc, (np.ndarray, torch.Tensor)):
        continue
    if tuple(cc.shape) != (5215, 192):
        continue
    cc = torch.as_tensor(cc, dtype=torch.float32)
    for activation in ("gelu", "silu", "relu", "tanh"):
        pred = post284(cc, u277, O277, activation)
        err = pred - zorig283
        scores284.append((
            float(torch.sqrt(torch.mean(err**2))),
            float(err.abs().max()), cname, activation
        ))
scores284.sort()
print("=== BEST ORIGINAL POSTERIOR RECONSTRUCTIONS ===")
for row in scores284[:10]:
    print(row)

orig_rms284, orig_max284, cname284, activation284 = scores284[0]
base_ctx284 = torch.as_tensor(
    globals()[cname284], dtype=torch.float32
).clone()

current_scores284 = []
for nprefix in ("out_norm", "core.out_norm"):
    if nprefix + ".weight" not in W282:
        continue
    cc = (
        base_ctx284 + delta_out282
        + norm277(Hflow281.mean(1), nprefix, W282)
        - norm277(hflow277.mean(1), nprefix, W282)
    )
    pred = post284(cc, u277, W282, activation284)
    err = pred - torch.from_numpy(S280["x"][:, 960:992])
    current_scores284.append((
        float(torch.sqrt(torch.mean(err**2))),
        float(err.abs().max()), nprefix, cc, pred
    ))
print("\n=== CURRENT CONTEXT NORMALIZATION CHECK ===")
for rms, mx, name, _, _ in current_scores284:
    print(name, "RMS", rms, "max", mx)
current_scores284.sort(key=lambda v: v[0])
zrms284, zmax284, normname284, ctx284, mu_pred284 = current_scores284[0]

Es284 = {}
if (orig_rms284 < 0.002 and orig_max284 < 0.02
        and zrms284 < 0.002 and zmax284 < 0.02):
    mu_actual284 = torch.from_numpy(S280["x"][:, 960:992])
    slots284 = Hflow281 + W282["core.slots"][None]
    raw284 = torch.cat((
        slots284, e277[:, None], u277[:, None],
        lin277(mu_actual284, "z_in", W282)[:, None]
    ), dim=1)
    E284 = prelude277(raw284, W282).detach().numpy().copy()
    for (drug, factor), uu in Ucf279.items():
        uu = torch.as_tensor(uu, dtype=torch.float32)
        # Anchor the negligible CPU/GPU discrepancy to the exported factual z.
        mucf = (
            post284(ctx284, uu, W282, activation284)
            - mu_pred284 + mu_actual284
        )
        rawcf = torch.cat((
            slots284, e277[:, None], uu[:, None],
            lin277(mucf, "z_in", W282)[:, None]
        ), dim=1)
        ecf = prelude277(rawcf, W282).detach().numpy().copy()
        Es284[(drug, factor)] = ecf
        j = 0 if drug == "insulin" else 1
        active = ds_rates[:, j] > 0
        print(drug, factor,
              "active contrast RMS",
              float(np.sqrt(np.mean((ecf[active] - E284[active])**2))),
              "inactive contrast RMS",
              float(np.sqrt(np.mean((ecf[~active] - E284[~active])**2))))
    print("SUCCESS: reconstructed factual and ten dose-edited feature sets.")
else:
    print("FAILED: no features fitted and no weights changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        f"\n## Run 284\nAudited cached contexts and posterior activations. "
        f"Best original: {cname284}/{activation284}, "
        f"RMS/max {orig_rms284:.6g}/{orig_max284:.6g}; "
        f"best current normalization {normname284}, "
        f"RMS/max {zrms284:.6g}/{zmax284:.6g}. "
        f"Built {len(Es284)} feature sets only if both checks passed. "
        f"No weights changed. Counts: 284 Python runs, 30 forwards.\n"
    )
```
