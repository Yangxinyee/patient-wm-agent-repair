The last run showed that the drug-atom builder in the kernel uses the weight dictionary, the code ids, and the fine-bin and z-score helpers. The 7.7 is probably a timing in seconds. My last few replies were withheld, so from here on I will print only names, shapes and summary statistics.

**Current plan.** The simulator's effects look linear and additive, so a counterfactual record can be built exactly:
- Scale the drug doses by m from the start hour.
- Shift the later readings by the estimated impulse response applied to the dose change.

I then teacher-force the replica on these records. That gives targets for a gated readout in a dead value-head unit: the edited model's prediction on a counterfactual history should equal its factual prediction plus the true shift.

**Next step.** Re-orient: list the kernel functions and the functions saved in /work/*.py, and inspect the bundle structure (types and lengths only). I want to know whether I can reuse a counterfactual-record builder or must write one.
- If cf.py and the kernel already hold a builder for scaled-dose bundles, I reuse it.
- If not, I write it next.

```python
import types, re
fn = sorted(k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print(len(fn), ' '.join(fn))
for f in ['cf.py','gen.py','heads.py','prep.py']:
    s=open('/work/'+f).read(); print(f, len(s), re.findall(r'^def (\w+)', s, flags=re.M))
s0 = sorted(BUN.keys())[0]; b = BUN[s0]
print('BUN', type(BUN).__name__, len(BUN), 'stay', s0, type(b).__name__, len(b))
print('elem', type(b[0]).__name__, len(b[0]) if hasattr(b[0],'__len__') else '', [type(x).__name__ for x in b[0]][:6])
x1 = b[0][1]; print('b[0][1]', type(x1).__name__, len(x1) if hasattr(x1,'__len__') else '', type(x1[0]).__name__ if hasattr(x1,'__len__') and len(x1) else '')
for nm in ['drug_atom','build_bundles','bvd','step','zpost','heads_mean']:
    if nm in globals(): c=globals()[nm].__code__; print(nm, c.co_varnames[:c.co_argcount])
```