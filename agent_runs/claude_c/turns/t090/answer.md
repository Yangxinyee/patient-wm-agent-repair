Run 80 printed the scale of the dose signal but then crashed. The dose direction `p` is nearly silent at reading bundles (sd 0.06) and strong at drug bundles (sd about 1.5), so it is selective. The crash came from my helper `in_ops_for`, which returned nothing.

The conversation was restarted, so I don't have the architecture details in front of me. Before building the z-space remap, I will reread ARCHITECTURE.md, list the current operations with their alphas, and list my kernel variables.

```python
print(open('/model/ARCHITECTURE.md').read())
print("OPS:", ops())
import types
print(sorted([k for k,v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType)])[:300])
```