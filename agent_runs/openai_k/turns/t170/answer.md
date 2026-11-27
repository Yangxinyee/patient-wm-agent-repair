The assay stopped because the readout weights were already stored as two 192-dimensional vectors, rather than a full feature vector. No weights changed, and this is an interface error rather than evidence against the approach. I’ll correct it and measure the existing directions’ drug-specific responses.

```python
import inspect, numpy as np, torch

dynamic_weights154 = {}
for ev155 in events151:
    view_weight155, interaction_weight155 = readoutweights151[ev155]
    m155 = models145[ev155]
    lv155 = lastvalues151[ev155]
    ratio155 = np.where(
        np.isfinite(lv155),
        (lv155-m155["median"])/m155["scale"], 0.
    )
    dynamic_weights154[ev155] = (
        np.asarray(view_weight155)[None, :]
        + ratio155[:, None] * np.asarray(interaction_weight155)[None, :]
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
print("original reconstruction RMS:",
      float(torch.sqrt(torch.mean(
          (out_original154-reference_out154)**2
      ))))

zero_weights154 = candidates90[0.0]
z_base154, token_base154 = posterior_token154(
    zero_weights154, context151["u"]
)
out_base154, h_base154 = local_token154(token_base154)
views_base154 = dynamic_views154(out_base154, h_base154)
print("zero-correction output RMS versus original:",
      float(torch.sqrt(torch.mean(
          (out_base154-out_original154)**2
      ))))

eps154 = 0.05
responses154 = {}
print("\n=== EXISTING DIRECTION RESPONSES ===")
print("event order:", events151)
for d155 in range(2):
    direction155 = torch.as_tensor(D89[:, d155], dtype=torch.float32)
    op155, hp155 = local_token154(
        token_base154 + eps154*direction155[None, :]
    )
    om155, hm155 = local_token154(
        token_base154 - eps154*direction155[None, :]
    )
    response155 = dynamic_delta154(
        dynamic_views154(op155, hp155),
        dynamic_views154(om155, hm155)
    ) / (2*eps154)
    responses154[d155] = response155
    own155 = np.asarray(drug151) == d155
    print("direction", d155,
          "norm", round(float(np.linalg.norm(D89[:, d155])), 4),
          "own-drug mean", np.round(response155[own155].mean(0), 4),
          "own-drug SD", np.round(response155[own155].std(0), 4))

print("\n=== ACTION HELPER ===")
f155 = rr35_scaled_action_encoding
print(inspect.signature(f155),
      "locals", f155.__code__.co_varnames,
      "globals", f155.__code__.co_names)

print("\n=== RETAINED ACTION-CONTRAST CONTAINERS ===")
found155 = 0
for n155, v155 in list(globals().items()):
    if n155.startswith("_") or not isinstance(v155, (list, tuple)):
        continue
    if (len(v155) == 2 and all(
        isinstance(c, dict) and {"drug", "indices", "u", "dc"} <= set(c)
        for c in v155
    )):
        print(n155, [
            (c["drug"], np.shape(c["indices"]), np.shape(c["u"]),
             np.quantile(c["dc"], [0, .5, 1]).round(4).tolist())
            for c in v155
        ])
        found155 += 1
        if found155 == 10:
            break

with open("/work/notes.md", "a") as f155:
    f155.write(
        "\n## Runs 154-155: repair derivative interface and assay existing directions\n"
        "- Run 154 stopped before computation: retained readoutweights151 already contains separate 192-dimensional view/interaction coefficients.\n"
        "- Corrected that interface and measured exact-core finite-difference responses; recovering physical action contrast helper.\n"
        "- No saved weights changed. Python runs: 155; forward passes: 10.\n"
    )
```