The architecture confirms that action edits affect shared state and potentially every output head. Earlier estimates suggest the baseline may understate insulin’s effect, but those estimates were provisional. I’ll recover the identification checks before choosing a target; agreement across independent record-based tests would strengthen that conclusion.

```python
from pathlib import Path
import numpy as np
import pandas as pd

nt = Path("/work/notes.md").read_text()
start = nt.find("## Causal identification")
print("NOTEBOOK LENGTH:", len(nt))
print(nt[start:start + 7200])
print("\nCACHED FUNCTIONS:",
      [k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\nCACHED DATA:")
for k, v in list(globals().items()):
    if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, v.shape)
```
