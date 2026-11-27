Restarted after the last reply was blocked; nothing in that reply ran. No edits are saved yet, and 3 of 50 forward passes are used. My working plan: make the core carry the insulin and norepinephrine rates in a clean channel, then read that channel at the value head through the unused (dead) hidden units. Before going on I'll check which helper functions and variables survive in the kernel. I'll also re-read the part of ARCHITECTURE.md that describes the head input (with_code, embed_transition, registers). This only gathers information, so there is nothing to confirm or refute yet.

```python
import types, re
user = {k: type(v).__name__ for k, v in globals().items()
        if not k.startswith('_') and not isinstance(v, types.ModuleType) and k not in ('In','Out','exit','quit','get_ipython')}
funcs = sorted([k for k, t in user.items() if t == 'function'])
others = sorted([f"{k}:{t}" for k, t in user.items() if t != 'function'])
print("FUNCS:", funcs)
print("VARS:", others[:150])
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
for kw in ['with_code', 'embed_transition', 'register', 'attr_heads.value']:
    for m in re.finditer(kw, arch):
        s = max(0, m.start()-200); print('----', kw, '\n', arch[s:m.start()+400]); break
```