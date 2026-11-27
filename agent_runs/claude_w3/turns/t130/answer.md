The previous output was refused for length, so I can't see what my last step produced. My notebook says the plan was to rebuild the model locally and check it against recorded states. Before going on, I'll take stock of the kernel variables and the files, printing only short summaries. I expect to find the architecture text, the loaded data arrays and possibly a partial reimplementation. If the reimplementation exists, I'll go straight to validating it.

```python
import os, json, types
skip = {'In','Out','exit','quit','get_ipython'}
vs = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and k not in skip and not isinstance(v, types.ModuleType)}
print(len(vs))
print(sorted(vs.items())[:200])
print(os.listdir('/work'), [os.listdir('/work/'+d) for d in os.listdir('/work') if d.startswith('fwd_')])
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
print([l for l in arch.split('\n') if l.startswith('#')])
```