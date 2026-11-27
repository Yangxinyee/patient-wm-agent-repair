The original replay agrees closely with the retained replay, and the route has two unused attention-output columns. The last cell stopped on a tuple-handling error before reporting route responses. I’ll fix that and measure responses from stopping through tripling each drug. Monotone, roughly proportional signals would support calibration; saturation or cross-drug leakage would weaken it.

```python
import numpy as np, torch

def means229(hh, oo, indices=None):
    result = readout214(hh, oo, indices=indices)
    if isinstance(result, tuple):
        choices = [x for x in result if hasattr(x, "shape") and len(x.shape) == 2
                   and x.shape[-1] == 4]
        if len(choices) != 1:
            raise RuntimeError(("Ambiguous readout", [getattr(x, "shape", None) for x in result]))
        result = choices[0]
    return np.asarray(result, dtype=float)

route_means229 = means229(route_h228, route_o228)
base_means229 = means229(check_h228, check_o228)
print("NEUTRAL ROUTE MEAN SHIFT", (route_means229 - base_means229).mean(0))
print("NEUTRAL ROUTE MEAN RMS", np.sqrt(np.mean((route_means229 - base_means229)**2, 0)))

def route_tokens229(candidate, action_inputs):
    retained = {k: W[k] for k in candidate}
    try:
        for k, v in candidate.items():
            W[k] = torch.as_tensor(v, dtype=torch.float32)
        u = torch.as_tensor(action_inputs, dtype=torch.float32)
        post = mlp136(torch.cat([context, e_cal161, u], 1), "z_post")[:, :32]
        h = torch.cat([a[0] for a in args_cache182], 0)
        dt = torch.cat([a[1] for a in args_cache182], 0)
        e = torch.cat([a[2] for a in args_cache182], 0)
        flowed = h * torch.exp(-dt[:, None, None] * torch.exp(W["core.flow.a"]))
        return torch.cat([flowed + W["core.slots"],
                          e[:, None, :], u[ids182, None, :],
                          linear136(post[ids182], "z_in")[:, None, :]], 1)
    finally:
        for k, v in retained.items():
            W[k] = v

def route_features229(candidate, action_inputs):
    result = att_features220(route_tokens229(candidate, action_inputs),
                             candidate["core.prelude.0.attn.q.weight"],
                             candidate["core.prelude.0.attn.kv.weight"])
    if isinstance(result, tuple):
        choices = [x for x in result if hasattr(x, "shape") and x.shape[-1] == 192]
        if len(choices) != 1:
            raise RuntimeError(("Ambiguous attention", [getattr(x, "shape", None) for x in result]))
        result = choices[0]
    return np.asarray(result)

features229 = route_features229(route228, Ubase215)
print("FEATURE SHAPE", features229.shape)
signal229 = features229[:, :4, response_cols228]
print("FACTUAL SIGNAL SD", signal229.std(axis=(0, 1)))
print("FACTUAL SIGNAL QUANTILES",
      np.quantile(signal229.reshape(-1, 2), [.01, .5, .99], axis=0))
print("SENSORS", list(causal_targets215["insulin"]), "rates",
      [(d, int(sum(drug_labels215 == d)),
        float(np.mean(rate_vector215[drug_labels215 == d])))
       for d in ["insulin", "norepinephrine"]])

local229 = {}
signals229 = {}
actions229 = {}
for di, drug in enumerate(["insulin", "norepinephrine"]):
    mask = drug_labels215 == drug
    for scale in [0., .5, 2., 3.]:
        scales = [1., 1.]
        scales[di] = scale
        uu = encode_actions168(tuple(scales))
        hh, oo = replay_route228(route228, uu)
        delta = means229(hh, oo) - route_means229
        local229[(drug, scale)] = delta
        actions229[(drug, scale)] = uu
        ss = route_features229(route228, uu)[:, :4, response_cols228]
        signals229[(drug, scale)] = ss
        per_rate = delta[mask].sum(0) / ((scale - 1.) * rate_vector215[mask].sum())
        signal_per_rate = ((ss - signal229)[mask].mean(1).sum(0)
                           / ((scale - 1.) * rate_vector215[mask].sum()))
        print(drug, scale, "sensor slopes", np.round(per_rate, 5),
              "transport slopes", np.round(signal_per_rate, 7))

with open("/work/notes.md", "a") as f229:
    f229.write("\n## Python 228–229 — repaired replay readout and dose-signal checks\n"
               "- Run228 stopped on tuple subtraction; repaired readout extraction without changing weights.\n"
               "- Original replay agreement is close (h RMS .00130, out RMS .000024).\n"
               "- Identified attention response columns 48 and 49 and tested local x0–x3 responses.\n"
               "- No response weights installed; current small neutral route still fails factual safeguard.\n")
```