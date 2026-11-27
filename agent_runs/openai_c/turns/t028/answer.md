Removing action input from the latent posterior barely changed factual predictions or rollouts, so that pathway is unlikely to explain the failure. Before attempting a more complex edit, I’ll test a sixteenfold numeric gain. It is useful only if factual NLL stays within budget and the dose responses grow substantially rather than saturating.

```python
import copy, numpy as np

repair_candidates = globals().get("repair_candidates", {})
repair_candidates["numeric_gain4"] = copy.deepcopy(ops())

print("Numeric dose axis:", numeric_dose_axis)
print("=== RETAINED DIRECTION OBJECTS ===")
for _name in ["head_surrogates", "whitened_dose_directions",
              "state_dose_coefficients", "dose_state_coeff",
              "state_C", "predicted_dose_states"]:
    if _name in globals():
        print(_name, compact_structure(globals()[_name]))

for _drug in ["insulin", "norepinephrine"]:
    print(_drug, {p: float(np.linalg.norm(v))
                  for p,v in dose_directions[_drug].items()})
print("=== OTHER POTENTIALLY USEFUL TENSORS ===")
for _name in NAMES:
    if any(s in _name for s in ["last_numeric", "embed_transition.value",
                               "query_norm", "z_head", "prelude.0.self_attn"]):
        print(_name, SHAPES[_name])

clear_ops()
rank1("embed.numeric.weight", numeric_dose_axis, 15.0, side="in")
print("Current operations:", ops())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 18: numeric gain 16\n"
            "- Latent-posterior action erasure had negligible factual and rollout effects (forwards 6–7).\n"
            "- Replaced numeric gain 4 with gain 16; no other edits.\n"
            "- Test whether stronger numeric signal remains factual-safe and avoids dose-response saturation.\n"
            "- Python runs: 18; forward passes: 10.\n")
```
