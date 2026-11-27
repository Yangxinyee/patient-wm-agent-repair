The retained paired readouts predict the current edit’s changes reasonably well, but that does not guarantee they will predict a new routing direction. I’ll first measure their response to the existing directions using the exact recurrent update. Consistent drug-specific signs and agreement with the official rollout would support using them for a new closed-form edit.

```python
import numpy as np, torch
import torch.nn.functional as F

torch.set_num_threads(8)

print("=== COMPACT INTERFACE CHECK ===")
print("local returns:", [tuple(x.shape) for x in original_result151])
print("next dt:", np.quantile(dt_next151, [0, .5, 1]))
for ev154 in events151:
    print(ev154, "readout parts:",
          [np.shape(x) for x in readoutweights151[ev154]])
for n154 in ("cf150", "c90"):
    if n154 in globals():
        print(n154, {
            k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
            for k, v in globals()[n154].items()
        })
print("dose helpers:", [
    n for n, v in list(globals().items())
    if callable(v) and hasattr(v, "__code__")
    and any(s in n.lower() for s in ("dose", "action", "counterfact", "cf_"))
][:25])

def posterior_token154(weights, u):
    x = torch.cat([context151["ctx"], context151["e"], u], dim=1)
    hidden = F.gelu(F.linear(
        x, weights["z_post.0.weight"], weights["z_post.0.bias"]
    ))
    z = F.linear(
        hidden, weights["z_post.2.weight"], weights["z_post.2.bias"]
    )[:, :32]
    token = F.linear(z, weights["z_in.weight"], weights["z_in.bias"])
    return z, token

base_lin154 = lin0

def local_token154(token, u=None):
    if u is None:
        u = context151["u"]
    def override154(x, prefix):
        if prefix == "z_in":
            return token
        return base_lin154(x, prefix)
    previous_lin154 = globals()["lin0"]
    globals()["lin0"] = override154
    try:
        result = repaired_core_local(
            context151["h"], context151["e"], u,
            torch.zeros((len(token), 32), dtype=torch.float32),
            context151["memory"], context151["mask"], context151["ages"]
        )
    finally:
        globals()["lin0"] = previous_lin154
    out = next(x for x in result if x.ndim == 2)
    h = next(x for x in result if x.ndim == 3)
    return out, h

def dynamic_views154(out, h):
    state = np.concatenate([
        out.numpy(), h.reshape(len(out), -1).numpy(),
        np.zeros((len(out), 32), dtype=np.float32)
    ], axis=1)
    return views144(state, dt_next151)

dynamic_weights154 = {}
for ev154 in events151:
    coef154 = np.asarray(readoutweights151[ev154][0])
    assert coef154.ndim == 1 and len(coef154) >= 384
    start154 = len(coef154) - 384
    m154 = models145[ev154]
    lv154 = lastvalues151[ev154]
    ratio154 = np.where(
        np.isfinite(lv154),
        (lv154 - m154["median"]) / m154["scale"], 0.
    )
    dynamic_weights154[ev154] = (
        coef154[start154:start154+192][None, :]
        + ratio154[:, None] * coef154[start154+192:][None, :]
    )

def dynamic_delta154(new_views, old_views):
    return np.column_stack([
        np.sum(
            (np.asarray(new_views[models145[ev]["variant"]])
             - np.asarray(old_views[models145[ev]["variant"]]))
            * dynamic_weights154[ev], axis=1
        )
        for ev in events151
    ])

z_original154, token_original154 = posterior_token154(
    originalweights151, context151["u"]
)
out_original154, h_original154 = local_token154(token_original154)
reference_out154 = next(x for x in original_result151 if x.ndim == 2)
print("\noriginal reconstruction RMS:",
      float(torch.sqrt(torch.mean(
          (out_original154-reference_out154)**2
      ))))

zero_weights154 = candidates90[0.0]
z_base154, token_base154 = posterior_token154(
    zero_weights154, context151["u"]
)
out_base154, h_base154 = local_token154(token_base154)
views_base154 = dynamic_views154(out_base154, h_base154)
print("zero-correction state RMS versus original:",
      float(torch.sqrt(torch.mean(
          (out_base154-out_original154)**2
      ))))

eps154 = 0.05
responses154 = {}
print("\n=== EXISTING TOKEN DIRECTION RESPONSES PER UNIT ===")
print("event order:", events151)
for d154 in range(2):
    direction154 = torch.as_tensor(D89[:, d154], dtype=torch.float32)
    op154, hp154 = local_token154(
        token_base154 + eps154 * direction154[None, :]
    )
    om154, hm154 = local_token154(
        token_base154 - eps154 * direction154[None, :]
    )
    response154 = dynamic_delta154(
        dynamic_views154(op154, hp154),
        dynamic_views154(om154, hm154)
    ) / (2 * eps154)
    responses154[d154] = response154
    own154 = np.asarray(drug151) == d154
    print("direction", d154,
          "norm", float(np.linalg.norm(D89[:, d154])),
          "own-drug mean", np.round(response154[own154].mean(0), 4),
          "own-drug SD", np.round(response154[own154].std(0), 4))

print("\n=== AVAILABLE COUNTERFACTUAL ACTION DIFFERENCES ===")
print("ucf151 differences by drug:",
      [(int(d), float(torch.norm(
          ucf151[np.asarray(drug151)==d]
          - context151["u"][np.asarray(drug151)==d]
      ))) for d in np.unique(drug151)])

with open("/work/notes.md", "a") as f154:
    f154.write(
        "\n## Run 154: exact recurrent finite-difference response assay\n"
        "- Implemented explicit posterior-token and exact local-update interfaces without gradients.\n"
        "- Reconstructed paired-surrogate state derivatives with fixed prior reading registers.\n"
        "- Tested existing token directions; no saved weights changed.\n"
        "- Python runs: 154; forward passes: 10.\n"
    )
```