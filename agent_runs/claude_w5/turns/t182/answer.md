The hooked core step `cstep_h` matches `cstep` exactly (max difference 0.0). I can now see h_new and inject a perturbation there.

What the injection needs:
- **Gate:** it must fire only at drug bundles.
- **Residualisation:** it must scale with the dose minus the policy-expected dose given the reading, otherwise factual predictions would be biased.

Which signal can carry that depends on whether a drug start and the reading that set its dose fall in the same bundle. If they do, both the reading and the dose are present in that one prelude step.

Next I will:
- print a compact bundle sequence for one stay;
- count, over the 100 stays, how often insulin and norepinephrine bundles also hold a glucose or MAP reading.

A high co-occurrence would support doing the residualisation inside the drug bundle.

```python
from collections import Counter
ab={'glucose_finger':'gf','glucose_lab':'gl','map_arterial':'ma','map_cuff':'mc','insulin':'I','norepinephrine':'N'}
def tag(a): return ab.get(a['code'],a['code'][:3])+('' if a['kind']==0 else '/k%d'%a['kind'])
print(' '.join('%.2f:%s'%(t,'+'.join(tag(a) for a in at)) for t,at in B[:22]))
C=Counter()
for s,BB in bundles_B.items():
    for t,at in BB:
        cs=set(a['code'] for a in at if a['kind']==0)
        hasI='insulin' in cs; hasN='norepinephrine' in cs
        hasG=bool(cs&{'glucose_finger','glucose_lab'}); hasM=bool(cs&{'map_arterial','map_cuff'})
        C[('I' if hasI else '')+('N' if hasN else '')+('G' if hasG else '')+('M' if hasM else '')]+=1
print(C.most_common(16))
```