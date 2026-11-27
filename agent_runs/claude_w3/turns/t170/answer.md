The last output grouped bundles by type and gave counts and error sizes for my local re-implementation of the model. Errors are small for most groups but large for some (above 1.0). So the re-implementation is still not exact enough to rely on.

My context was reset. Before writing any edit I'll re-read the architecture description and list the kernel's variables, so I know what code and data I already have. If a validated per-step function already exists, I'll reuse it. If not, I'll fix the parts with large errors.

```python
import os
print(open('/model/ARCHITECTURE.md').read()[:6000])
print([k for k,v in globals().items() if not k.startswith('_') and not callable(v)][:80])
print([k for k,v in globals().items() if not k.startswith('_') and callable(v)][:80])
```