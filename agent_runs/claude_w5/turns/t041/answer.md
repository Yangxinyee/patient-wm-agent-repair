I'm picking up after a restart. The last run confirmed that the 100 stays I rebuilt match the tool's output on all 100, with 0 mismatches. My notebook says the original model barely responds to doses. For insulin x2, the model's shift is about -0.5 mg/dL, while my g-estimate of the true effect is about -14. To design the weight edit I need the exact architecture again: where actions enter, and how the z posterior and prelude token feed the value heads. So first I'll re-read ARCHITECTURE.md, the full notebook, what fwd_2 holds, and which variables my kernel still has.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====NOTES=====")
print(open('/work/notes.md').read())
for d in ['fwd_1','fwd_2']:
    p='/work/'+d
    print(d, os.listdir(p) if os.path.exists(p) else None)
print([k for k in globals().keys() if not k.startswith('_')][:300])
```