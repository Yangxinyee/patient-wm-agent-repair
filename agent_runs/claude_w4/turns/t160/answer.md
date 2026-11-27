The last run compared four versions of a fit, labelled lnM, ageKV, noage and lnM_ageKV. The ageKV version fit far better than the others: an error of 39, against 79 to 253, with R² of 0.998 and 0.994. So the age-dependent key/value term is the part that matters.

My notebook stops at run 122, though. Runs 123 to 139, including what this fit was for, are not written down. Before I build anything on that result, I need to rebuild my picture of the work.

Next I will:
- re-read the architecture description, since my plan depends on how the prelude and state update work;
- list the variables and functions in the kernel, in the order they were defined, so the newest ones show what runs 123 to 139 were doing.

What would confirm my current hypothesis: kernel objects that show runs 123 to 139 built a CPU model of the prelude's head-1 value path or the FFN path, and of how a change added there carries over to later bundles. If the objects show something else, I will re-plan from the notebook's last stated plan.

```python
import types
print(open('/model/ARCHITECTURE.md').read())
items = [(k, v) for k, v in list(globals().items()) if not k.startswith('_')]
desc = []
for k, v in items:
    if isinstance(v, types.ModuleType) or k in ('In', 'Out', 'exit', 'quit', 'get_ipython'):
        continue
    t = type(v).__name__
    if hasattr(v, 'shape'):
        s = str(tuple(v.shape))
    elif isinstance(v, (list, tuple, dict, set)):
        s = f'len{len(v)}'
    elif isinstance(v, (int, float)):
        s = repr(v)[:12]
    elif isinstance(v, str):
        s = repr(v)[:24]
    else:
        s = ''
    desc.append(f'{k}<{t}{":" + s if s else ""}>')
print(len(desc), 'globals')
print(' '.join(desc))
```