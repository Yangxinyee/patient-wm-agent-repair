lib3.py holds experiment helpers rather than a finished edit: a scaffold, mean-ablation of units, `ds_edit` (re-encode a dataset with scaled doses) and `bundle_rates`. So no edit exists yet.

Before committing to a design I need one architecture fact. Is the state-update gate (`delta_in` → `abar`) input-dependent and per-dimension? If it is, a single state dimension can be overwritten at insulin bundles and reset at glucose-reading bundles. It would then hold "insulin since the last glucose reading". That is much simpler than a decaying memory with regression-based compensation.

- **Confirms:** a `delta_in` weight of shape (192, 192), or similar per-dim, applied to E.
- **Refutes:** a scalar or per-slot delta, in which case I fall back to the decay-only design.

```python
import re
A=open('/model/ARCHITECTURE.md').read()
for kw in ['delta_in','a_in','b_in','abar','flow.a','c_out']:
    for m in re.finditer(re.escape(kw),A):
        s=A[max(0,m.start()-120):m.end()+160].replace('\n',' ')
        print('['+kw+']',s[:280]); break
for n in NAMES:
    if any(k in n for k in ['delta','a_in','b_in','flow','c_out','transition','value_bin']):
        print(n,SHAPES[n])
```