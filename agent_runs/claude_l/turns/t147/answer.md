I'm resuming after the restart. Run 141 showed that my replica of the z posterior is still off: the median error is about 0.29 whichever activation I use. The encoder `e` and the action encoder `u` match exactly, so the mismatch must come from how the z_post input is put together or from the context it reads. I no longer have ARCHITECTURE.md or my replica code in view, so I'll re-read them and list the kernel variables before spending more runs. Getting the z_post wiring right matters because my planned edit acts there: it should make predictions respond to the dose residual (dose minus what the state predicts), with the true decaying effect over time. If I find a z_post or core detail I implemented differently, that confirms the replica bug is in the wiring. If the document matches my code exactly, the bug is in the inputs (ctx), not the formula.

```python
import os, re, json
print(open('/model/ARCHITECTURE.md').read())
print("=========== wm.py ===========")
src = open('/work/wm.py').read()
print(len(src)); print(src[:6000])
print("=========== globals ===========")
g = [k for k,v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, type(os))]
print(sorted(g)[:300])
print([k for k,v in globals().items() if callable(v) and not k.startswith('_')][:200])
```