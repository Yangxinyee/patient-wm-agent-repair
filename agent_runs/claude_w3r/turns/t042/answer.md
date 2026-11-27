The last run compared my local re-implementation of the model with the real model states, trying each combination of the two unit codes (mmHg, mg/dL). Every combination left a large error: median about 0.35–0.43 and maximum up to about 0.97. So the unit codes are not the cause; the mismatch is somewhere else in my re-implementation. Because the conversation restarted, I will first re-read the architecture description, my saved implementation and the list of kernel variables before debugging further. If an architecture detail is missing from impl.py, that would explain the error. If impl.py follows the description exactly, the bug is in how I fed the inputs.

```python
import os, re
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
print(arch[:12000])
```