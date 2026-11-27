The last run checked my local copy of the value heads against the model's factual predicted means. It is not exact yet: the median gap is 2.46 mg/dL when z is set to the prior mean. That error may come from my implementation, or from using the prior mean of z instead of what the model actually uses.

This is a restarted conversation, so I need the architecture details again before I design a head-level or core edit. Next I will reread ARCHITECTURE.md, see what kernel variables and helper files exist, and log the restart in the notebook. If the document says value prediction uses a sampled z or the posterior z, that would explain the mismatch.

```python
import os, re
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:12000])
with open('/work/notes.md','a') as f:
    f.write("\n## Session 3 (restart 3)\n- Run 77: local head impl vs factual pred mean (z=prior mean): median abs err 2.46 -> not exact yet.\n")
print([k for k in globals().keys() if not k.startswith('_')][:200])
```