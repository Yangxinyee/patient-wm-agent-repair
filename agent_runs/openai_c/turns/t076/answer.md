Numeric gain 4 is the only retained edit with clear rollout benefit, and it leaves measurement inputs unchanged. I’ll test gain 16 on that same drug-numeric direction. A substantially stronger response with safe factual loss would support this pathway; saturation or excessive loss would rule out further simple amplification.

```python
from pathlib import Path
import json
import numpy as np

notes = Path("/work/notes.md").read_text()
print("EARLIER GAIN TESTS\n", notes[12000:15000])
for k in [6, 7, 9, 10, 11, 12]:
    summary = json.loads(Path(f"/work/fwd_{k}/summary.json").read_text())
    print("FORWARD", k, "RESULTS", summary.get("results"))

dose_numeric_direction = np.array([0., 0., 1., 0., 1., 0.]) / np.sqrt(2.)
clear_ops()
rank1("embed.numeric.weight", dose_numeric_direction, 15.0, side="in")
repair_candidates["numeric_gain16"] = [
    {"op": "rank1", "name": "embed.numeric.weight",
     "vector": dose_numeric_direction.copy(), "alpha": 15.0, "side": "in"}
]
print("INSTALLED", ops())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 52: stronger drug-numeric gain\n"
            "- Installed gain 16 on the joint dose/rate numeric direction; measurement numeric inputs unchanged.\n"
            "- Numeric gain 4 remains saved for rollback.\n"
            "- Hypothesis: larger drug-numeric signal strengthens causal rollout responses. "
            "Must check factual safety and saturation.\n"
            "- Forward passes used: 22.\n")
```
