This is a restarted conversation, so I need my context back before going further. Run 37 compared my local re-implementation of the forward pass against the tool's outputs under three activation choices. ReLU matched best (corr 0.976, slope about 1.02). The candidate and target vectors still differ noticeably, so my local replica is close but not exact. I don't remember the architecture details or which kernel variables I built.

Next, I will re-read ARCHITECTURE.md and list the variables in the kernel. This tells me what I have, so I can finish the local replica and work out where dose information enters the model. The check: if the architecture text explains the residual mismatch, I can fix the replica; if not, I'll rely more on the forward tool.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====")
print([k for k,v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, type(os))][:200])
```