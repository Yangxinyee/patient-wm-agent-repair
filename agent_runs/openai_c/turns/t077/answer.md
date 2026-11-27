The recovered notebook shows gain 16 was already tested: it saturated and performed worse than gain 4. I won’t spend another pass on it. I’ll instead amplify the normalized state contribution to the reading head by 50%, testing whether the head—not the drug input—is limiting the response. Factual loss will determine whether this is usable.

```python
from pathlib import Path
import numpy as np

clear_ops()
rank1("embed.numeric.weight", dose_numeric_direction, 3.0, side="in")
scale("query_norm.weight", 1.5)

repair_candidates["query_gain15"] = [
    {"op": "rank1", "name": "embed.numeric.weight",
     "vector": dose_numeric_direction.copy(), "alpha": 3.0, "side": "in"},
    {"op": "scale", "name": "query_norm.weight", "c": 1.5}
]
print("INSTALLED", ops())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 53: query-state amplification pilot\n"
            "- Recovered existing gain-16 results: factual +0.00246, insulin -2.231/+1.531, "
            "NE +0.157/+0.085. No new forward pass spent; restored numeric gain 4.\n"
            "- Installed query_norm.weight scale 1.5, leaving its bias and reading registers unchanged.\n"
            "- Hypothesis: stronger state contribution at the reading head increases dose sensitivity. "
            "Factual safety is untested.\n"
            "- Forward passes used: 22.\n")
```
