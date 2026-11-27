Suppressing the lab gate reduced cached NLL cost, but the stronger corrections still have a cost on finger readings. I also reconstructed the separator rather than reusing the earlier probe, so I’ll check its exact coefficients before a GPU trial. I’ll compare event-specific costs and intervention extrapolation; a stable finger gate with negligible lab and MAP correction is the required result.

```python
import numpy as np, torch

print("STORED SEPARATOR COEFFICIENTS")
for lam, coef in gate_sep_candidates.items():
    a = np.asarray(coef)
    print(lam, a.shape, "norm", float(np.linalg.norm(a)),
          "last entries", np.round(a.reshape(-1)[-5:], 5))

separator_options = {"reconstructed": (selective_sep_w.copy(), selective_sep_b)}
coef = np.asarray(gate_sep_candidates[0.01])
if coef.shape == (193,):
    separator_options["stored_raw"] = (coef[:192], float(coef[192]))
if coef.shape == (152,):
    separator_options["stored_sc_Q"] = (sc_Q @ coef[:151], float(coef[151]))
if coef.shape == (151,):
    separator_options["stored_r3_Q"] = (r3_Q @ coef[:150], float(coef[150]))

print("\nSEPARATOR COMPARISON, ALL LATENT DRAWS")
for label, (w, b) in separator_options.items():
    c = sc_fact["tuning"]
    scores = c["x"] @ w + b
    print(label)
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        mask = c["meta"].event.to_numpy() == event
        print(event, np.round(np.quantile(scores[:, mask], [.01, .5, .99]), 4))

def cache_losses(cache, w, b, scale):
    score = cache["x"] @ np.asarray(w, np.float32) + np.float32(b)
    gt = torch.from_numpy(cache["gate"])
    tilt = (torch.nn.functional.gelu(gt + torch.from_numpy(score))
            - torch.nn.functional.gelu(gt)).numpy() * np.float32(scale)
    logits = cache["logits"] + tilt[..., None] * sc_rank
    p = torch.softmax(torch.from_numpy(logits), -1).mean(0).numpy()
    return -np.log(np.maximum(p[np.arange(len(cache["bins"])), cache["bins"]], 1e-30))

print("\nEVENT-SPECIFIC CACHED COST")
candidate = r3_candidates[(.1, 1.)]
for label, (w, b) in separator_options.items():
    selective_sep_w, selective_sep_b = np.asarray(w), float(b)
    for scale, strength in [(2., 0.), (2., 30.), (3., 30.)]:
        c0 = sc_fact["tuning"]
        c = selective_cache(c0, strength)
        loss = cache_losses(c, candidate["w"], candidate["b"], scale)
        p0 = torch.softmax(torch.from_numpy(c0["logits"]), -1).mean(0).numpy()
        loss0 = -np.log(np.maximum(p0[np.arange(len(c0["bins"])), c0["bins"]], 1e-30))
        print(label, "scale", scale, "gate strength", strength,
              {e: round(float((loss-loss0)[c0["meta"].event.to_numpy() == e].mean()), 6)
               for e in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]})

print("\nSYNTHETIC EVENT COUNTS")
print(sc_syn[1.0]["meta"].event.value_counts().to_dict())
print("\nFIRST TUNING STAY IDS")
print([int(s) for s in sc_fact["tuning"]["meta"].stay_id.unique()])
```
