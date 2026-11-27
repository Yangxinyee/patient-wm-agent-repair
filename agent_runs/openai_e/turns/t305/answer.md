The trial failed factual fit: NLL rose by 0.02294 nats, well above the limit. The small local estimate therefore understated its full-model disturbance. I’ll reduce the nonlinear edit to 15% of its trial strength and inspect where the loss arose. If the excess loss falls sharply, a weaker version may still be useful; otherwise I’ll abandon this route.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== FORWARD 30 COLUMNS ===")
for _model in ["original", "edited"]:
    _df = pd.read_csv(f"/work/fwd_30/factual_{_model}.csv")
    print(_model, list(_df.columns), _df.head(2).to_dict("records"))
print("=== PRIOR NONLINEAR / CALIBRATION OUTCOMES ===")
import re
_sections = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
for _sec in _sections:
    if re.search(r"## Run (94|95|96|97|98|153|154|155|156|157|176|177|178|179|180|181|183|184|185|186|187|188|231|232|233|234)\b", _sec):
        print(_sec[:420])

# Reject full trial; install a substantially smaller version from original weights.
mult263 = 0.15
save_edit("core.prelude.0.ffn.2.weight",
          np.asarray(load_original("core.prelude.0.ffn.2.weight")) +
          (mult263 * delta_w262).astype(np.float32))
save_edit("core.prelude.0.ffn.2.bias",
          np.asarray(load_original("core.prelude.0.ffn.2.bias")) +
          (mult263 * delta_b262).astype(np.float32))
print("nonlinear scale multiplier", mult263,
      "absolute scale", mult263 * scale262,
      "edited", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 263: reject full nonlinear trial; test reduced strength\n"
             "Forward30 factual increase=.02294 on tuning100..199, unacceptable. "
             "Local anchored estimates missed full-model disturbance. "
             "Reduced prelude trial to 15% strength, retaining joint236 b_in. "
             "This version is unaccepted pending paired validation. "
             "Python runs: 263; forwards: 30.\n")
```