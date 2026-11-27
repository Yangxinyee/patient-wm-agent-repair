The clean insulin residual (exact-integer cells, split readings excluded) is exogenous: it does not depend on the previous reading, the previous dose or the lab. Its effects on finger glucose are null at every lag from 1 to 12 hours, with |b| ≤ 0.13 and lag 1 at +0.01 ± 0.04. A shock also does not carry into later doses.

The implied shifts are all consistent with zero:
- insulin x2 on finger glucose: +0.31 mg/dL (SE 0.22–0.73);
- insulin x1/2: −0.16;
- the lab outcome: +0.10 (SE 0.41–1.37).

So both drugs have no causal effect, and the edit's exact-zero answers are right. The remaining risk is technical: making sure the model's insensitivity to dose holds for every dose the evaluation could produce. I set rate/dose bin rows 1–16 to row 8 but left row 0 alone. If row 0 is the bin for values below the lowest edge, a halved very small rate could land there, and the model would then react to the change.

Next I'll inspect the bin definitions in VOCAB and ARCHITECTURE.md:
- Row 0 is padding/missing, or no halved or doubled rate can reach it: the edit is complete.
- A drug value can fall into row 0: I'll neutralise row 0 too.

```python
arch=open('/model/ARCHITECTURE.md').read()
import re
for kw in ['bins','rate','dose','numeric','z_','edges','centre','center']:
    idxs=[m.start() for m in re.finditer(kw,arch)][:6]
    for i in idxs[:3]:
        print('['+kw+']', arch[max(0,i-200):i+300].replace('\n',' | '))
        print('---')
print(list(VOCAB.keys()))
for k in VOCAB:
    if any(s in k.lower() for s in ['rate','dose','bin','edge','norm']):
        print(k, str(VOCAB[k])[:800])
```