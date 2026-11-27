The stored separator is register-protected and remains stable on the synthetic dose edits, which is encouraging. The run failed before changing any weights because the snapshot used a NumPy copy method on a Torch tensor. I’ll fix that and install the same reversible trial; GPU factual and rollout results will determine whether it is useful.

```python
import numpy as np, torch

selective_sep_w, selective_sep_b = protected_separators[0.01]
selective_strength = 30.0
selective_trial_scale = 3.0
selective_trial_lam = 0.1
selective_trial_before = {n: load(n).clone() for n in sc_names}

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

print("CACHED TRIAL NLL DELTAS",
      {s: round(sc_evaluate(selective_cache(cache, selective_strength),
                           c["w"], c["b"], selective_trial_scale)["nll"]
                - cache["original_nll"], 6)
       for s, cache in sc_fact.items()})

for name, tensor in selective_trial.items():
    save_edit(name, torch.from_numpy(np.asarray(tensor, dtype=np.float32)).clone())
print("INSTALLED", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Reversible finger-selective trial; Python 196, forwards 16/50\n")
    f.write("- Python 195 failed before weight changes; corrected Torch snapshot method.\n")
    f.write("- Installed register-protected separator lambda 0.01, gate strength 30; r3 lambda 0.1, scale 3.\n")
    f.write("- Synthetic separator scores remain approximately +/-0.05 under insulin edits.\n")
    f.write("- NOT accepted until paired GPU factual and rollout validation; scale2_checkpoint retained.\n")
```
