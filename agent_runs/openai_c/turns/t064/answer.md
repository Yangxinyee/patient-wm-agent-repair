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
