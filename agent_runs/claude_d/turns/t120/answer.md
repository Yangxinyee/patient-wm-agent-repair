Run 107 printed error numbers from some variant comparison, and I no longer have its context. I can't tell exactly what it measured. The notebook says out-from-h is reproduced exactly in numpy, but the recurrent core was never reproduced exactly. My working plan is a correction at the value head that is centred on the dose the protocol would have expected. I would then tune its scale with dose_edit passes.

Before designing that edit I need two things. First, which variables survive in the kernel. Second, a refresher on the architecture's head and recurrence details: what feeds the value head, whether `out` feeds back into the recurrence, and how NLL is computed. Below I list the kernel variables and the section headers of ARCHITECTURE.md, keeping the output short.

```python
import types, re
g = {k:type(v).__name__ for k,v in globals().items() if not k.startswith('_') and not isinstance(v,(types.ModuleType,)) and not callable(v)}
print(len(g)); print(', '.join(f"{k}:{t}" for k,t in sorted(g.items()))[:3000])
fn = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and isinstance(v, types.FunctionType)]
print('FUNCS:', ', '.join(sorted(fn))[:1500])
A = open('/model/ARCHITECTURE.md').read()
print(len(A)); print([l for l in A.splitlines() if l.startswith('#')])
```