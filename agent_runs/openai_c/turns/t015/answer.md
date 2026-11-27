The inspection confirmed the posterior’s 576-input first layer, but the printout failed before applying any edits because the numeric vocabulary is a list. No hypothesis has been tested yet. I’ll apply the intended action-block removal now, then check factual fit before interpreting any rollout changes.

```python
import copy
import numpy as np
from pathlib import Path

assert SHAPES["z_post.0.weight"] == (192, 576)
clear_ops()
for j in range(384, 576):
    direction = np.zeros(576, dtype=np.float32)
    direction[j] = 1.0
    rank1("z_post.0.weight", direction, -1.0, side="in")

saved_candidates["B_posterior_action_erasure"] = copy.deepcopy(ops())
print("Candidate B: removed z_post.0.weight action columns 384:576.")
print("Direct action encoder and prelude action token unchanged.")
print("Operation count:", len(ops()))
print("Numeric vocabulary type:", type(VOCAB["numeric"]).__name__)
print("Numeric vocabulary:", str(VOCAB["numeric"])[:3500])

with Path("/work/notes.md").open("a") as f:
    f.write("""
## Runs 9–10: candidate B
- Run 9 inspection confirmed z_post.0.weight is (192,576), then failed during an optional vocabulary print before any edit.
- Run 10 erased its action-input block: last 192 columns of [context, observation, action].
- Context/observation inputs, direct prelude action token, measurement heads and numeric embeddings unchanged.
- Hypothesis: action-dependent latent inference counteracts direct treatment dynamics. This requires experimental validation.
- Python runs: 10; forward passes: 5.
""")
```