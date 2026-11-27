The saved states show dose-related changes in the latent state, and the model conditions that latent update directly on the current action. I’ll test whether this route counteracts the direct treatment pathway by removing only the action inputs to the posterior. Improved dose responses with little factual loss would support that hypothesis; otherwise I’ll undo it.

```python
import copy
import numpy as np
from pathlib import Path

print("=== LATENT AND FLOW TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_prior", "z_in", "z_head",
                             "flow", "a_in", "delta_in", "monitor")):
        print(n, SHAPES[n])

print("\n=== DRUG NUMERIC ENCODING ===")
for drug_name in ("insulin", "norepinephrine"):
    print(drug_name, VOCAB["numeric"].get(drug_name, "not keyed by name"))

posterior_input_names = [
    n for n in NAMES
    if "z_post" in n and len(SHAPES[n]) == 2 and SHAPES[n][1] == 576
]
assert len(posterior_input_names) == 1, posterior_input_names
posterior_input_name = posterior_input_names[0]

clear_ops()
for j in range(384, 576):
    direction = np.zeros(576, dtype=np.float32)
    direction[j] = 1.0
    rank1(posterior_input_name, direction, -1.0, side="in")

saved_candidates["B_posterior_action_erasure"] = copy.deepcopy(ops())
print("\n=== CANDIDATE B ===")
print("Removed action columns 384:576 of", posterior_input_name)
print("Direct action encoder and prelude action token unchanged.")
print("Operation count:", len(ops()))

with Path("/work/notes.md").open("a") as f:
    f.write(f"""
## Run 9: candidate B
- Hypothesis: action-dependent posterior inference counteracts the direct treatment pathway.
- Erased only the action-input block (last 192 of 576 columns) of {posterior_input_name}.
- Context, observation input, direct action token, measurement heads and numeric embeddings unchanged.
- This is a structural hypothesis, not an established causal interpretation; paired factual and dose tests will decide whether to retain it.
- Python runs: 9; forward passes: 5.
""")
```