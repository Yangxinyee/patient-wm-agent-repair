The recovered tests show that both the slot shear and latent-input amplification gave little benefit. One untested possibility is that the prior-latent reading contribution masks the direct state’s drug signal. I’ll disable that contribution, leaving its bias fixed. It is useful only if factual fit survives and dose responses improve.

```python
import json
from pathlib import Path

# Keep the established numeric-dose edit and remove only the prior-z
# contribution to the reading view. No recurrent latent update is changed.
replay_ops(repair_candidates["numeric_gain4"])
scale("z_head.weight", 0.0)

repair_candidates["no_prior_head"] = freeze_ops(ops())
print("Installed operations:", ops())

with open("/work/notes.md", "a") as _notes:
    _notes.write(
        "\n## Python run 66: prior-head ablation\n"
        "- Installed numeric gain 4 plus z_head.weight x0; z_head.bias and recurrent latent pathways unchanged.\n"
        "- Hypothesis: prior-latent reading contribution masks a more useful direct recurrent dose response.\n"
        "- This is an exploratory pathway test, not an established causal correction.\n"
        "- Require factual increase <=0.005, then assess all four dose rollouts.\n"
        "- Forward passes used: 25.\n"
    )
```