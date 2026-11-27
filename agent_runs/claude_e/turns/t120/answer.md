The candidate edit `ED` matches the plan in my notes and changes only the value head:

- **Repurposed units.** Four hidden units of the 192-unit MLP (26, 41, 131, 159) get new input weights, biases and output weights. These are presumably one code-gated unit per code: finger and lab glucose for insulin, arterial and cuff MAP for norepinephrine.
- **Transition table.** The rows for codes 5, 6, 7 and 10, which are indexed by the last bin, are changed. Most likely they remove the mean of the correction within each last-bin cell. That would make the correction act on the dose minus the dose expected for the current glucose tier, not on the raw dose.

The readouts are weak. The insulin readout has CV R² 0.40, and its slope on the true deviation is 0.40. The norepinephrine readout has R² 0.17 and slope 0.18. So a unit-gain correction has to divide by those slopes. That is where the factual-NLL cost will come from.

Before using `make_edit(a_ins, a_ne, K)` I need to know what it does. I will print the globals, attributes and constants it and the other helpers reference. This is a cheap substitute for the lost source code. Then I can check what ED was built with and evaluate it offline.

```python
import dis
def refs(f):
    c=f.__code__
    inner=[x for x in c.co_consts if hasattr(x,'co_code')]
    names=list(c.co_names)+[n for ic in inner for n in ic.co_names]
    consts=[x for x in c.co_consts if not hasattr(x,'co_code') and x is not None]
    return c.co_varnames, names, consts
for fnm in ['make_edit','head_run','fit_readout','refit','vref']:
    v,n,k=refs(globals()[fnm])
    print('==',fnm,'\n vars',v,'\n names',n,'\n consts',[str(x)[:40] for x in k])
```