The saved readout fits are sensitive to state changes, but the diagonal edits require large parameter changes and remain untested. I’ll recover the architecture and causal estimates before choosing another edit. My hypothesis is that a targeted change can strengthen the record-supported dose response; rollout improvement with factual NLL within the limit would confirm it.

```python
from pathlib import Path
import numpy as np
print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:15000])
print("\nPERSISTENT VARIABLE NAMES")
print(sorted(k for k in globals() if not k.startswith("_")))
print("\nCURRENT OPERATIONS")
print(ops())
```
