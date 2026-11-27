The cached event losses show that lab readings contribute much of the current NLL cost, and the selective gate removes that contribution. Finger loss still limits the correction strength. I’ll install a reversible, finger-selective scale-3 trial using the protected separator. It is acceptable only if GPU factual NLL stays within +0.005 and the insulin rollout moves closer to the record-derived target.

```python
import numpy as np, torch

train_x = np.concatenate([sc_fact[s]["x"].mean(0) for s in ["first", "extra"]])
train_e = np.concatenate([sc_fact[s]["meta"].event.to_numpy() for s in ["first", "extra"]])
train_keep = np.isin(train_e, ["glucose_finger", "glucose_lab"])
train_target = -(train_e[train_keep] == "glucose_lab").astype(float)

print("PROTECTED SEPARATOR CHECK")
protected_separators = {}
for lam, w in gate_sep_candidates.items():
    w = np.asarray(w, dtype=np.float64)
    b = float(np.mean(train_target - train_x[train_keep] @ w))
    protected_separators[lam] = (w, b)
    c = sc_fact["tuning"]
    score = c["x"] @ w + b
    syn_score = sc_syn[2.0]["x"] @ w + b
    print(lam, "protected residual",
          round(float(np.linalg.norm(w - sc_Q @ (sc_Q.T @ w))), 6),
          "finger", np.round(np.quantile(score[:, c["meta"].event == "glucose_finger"],
                                         [.01, .5, .99]), 4),
          "lab", np.round(np.quantile(score[:, c["meta"].event == "glucose_lab"],
                                      [.01, .5, .99]), 4),
          "synthetic", np.round(np.quantile(syn_score, [.001, .5, .999]), 4))

selective_sep_w, selective_sep_b = protected_separators[0.01]
selective_strength = 30.0
selective_trial_scale = 3.0
selective_trial_lam = 0.1
selective_trial_before = {n: load(n).copy() for n in sc_names}

def make_selective_contrast_graft(w, b, scale, gate_strength=30.0):
    out = make_contrast_graft(w, b, scale)
    n0, n1 = map(int, sc_nodes)
    gw = sc_gate_w + gate_strength * selective_sep_w
    gb = sc_gate_b + gate_strength * selective_sep_b
    out[sc_names[0]][n0] = np.asarray(gw + w, np.float32)
    out[sc_names[0]][n1] = np.asarray(gw, np.float32)
    out[sc_names[1]][n0] = np.float32(gb + b)
    out[sc_names[1]][n1] = np.float32(gb)
    return out

c = r3_candidates[(selective_trial_lam, 1.0)]
selective_trial = make_selective_contrast_graft(
    c["w"], c["b"], selective_trial_scale, selective_strength
)
print("\nCACHED TRIAL NLL DELTAS",
      {s: round(sc_evaluate(selective_cache(cache, selective_strength),
                           c["w"], c["b"], selective_trial_scale)["nll"]
                - cache["original_nll"], 6)
       for s, cache in sc_fact.items()})

for name, tensor in selective_trial.items():
    save_edit(name, tensor)
print("INSTALLED", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Reversible finger-selective trial; Python 195, forwards 16/50\n")
    f.write("- Installed protected-separator gate strength 30, r3 lambda 0.1, scale 3.\n")
    f.write("- Current trial is NOT accepted: paired GPU factual and rollout checks required.\n")
    f.write("- scale2_checkpoint and selective_trial_before preserve the prior GPU-validated fallback.\n")
```
