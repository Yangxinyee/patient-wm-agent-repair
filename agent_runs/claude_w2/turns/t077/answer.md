The last run compared my head re-implementation with the forward tool, per code.

**What run 71 showed.** For each code, my re-implementation gives a value NLL above the tool's:

| Code | Mine | Tool |
|---|---|---|
| glucose_finger | 1.547 | 1.526 |
| glucose_lab | 1.251 | 1.214 |
| map_arterial | 1.335 | 1.264 |
| map_cuff | 1.620 | 1.556 |

So the re-implementation is not exact yet. The likely cause is that some head input is not built as the model builds it. Candidates are the z term (the states give the posterior mean, but the head uses a prior-based z) and the flowed h term.

**Where things stand.** The records give clear dose effects in this simulated ICU. On sel1 they predict about −14 for insulin x2 and +4.7 for norepinephrine x2. The original model gives only about −2.1 and +1.1. Even teacher-forced, the model under-uses the dose deviation inside a band. So a correction tied to the dose deviation (dose minus what the policy would give at the current glucose) should both fix the rollouts and improve the factual NLL.

**Next step.** The conversation restarted, so I first recover my working state: which variables exist in the kernel, and the code of my last runs. If the kernel keeps an input history, I can read back my head re-implementation and find the mismatch. I will also check whether a full local re-implementation of the forward pass is within reach. That would let me test edits on the local train records without spending forward passes.

**What would confirm or refute this.** If the history shows the head uses a prior-based z where I fed the posterior mean, that explains the gap. If not, I re-read the architecture document's head section.

```python
import types, json
skip = {'In','Out','exit','quit','get_ipython'}
ns = {}
for k, v in list(globals().items()):
    if k.startswith('_') or k in skip or isinstance(v, types.ModuleType) or callable(v) and not hasattr(v,'shape'):
        continue
    s = getattr(v, 'shape', None)
    if s is None and hasattr(v, '__len__'):
        try: s = ('len', len(v))
        except Exception: s = None
    ns[k] = (type(v).__name__, s if not hasattr(s,'__iter__') or isinstance(s,tuple) else tuple(s))
print(len(ns)); print(ns)
fns = [k for k, v in globals().items() if callable(v) and not k.startswith('_') and isinstance(v, types.FunctionType)]
print('functions:', fns)
try:
    hist = In
    print('history len', len(hist))
    for i in range(max(1, len(hist)-3), len(hist)):
        print('=' * 20, 'In[%d]' % i, '=' * 20)
        print(hist[i][:5000])
except NameError:
    print('no In history')
```