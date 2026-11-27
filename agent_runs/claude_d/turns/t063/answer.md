The notebook shows where things stand. The training records give insulin and norepinephrine effects that are linear and additive. The model's rollouts underestimate the insulin effect about 8x and get the norepinephrine sign wrong. The targets on the sel1 stays are about -14.0 (insulin x2), +7.0 (insulin x1/2), +4.85 (NE x2) and -2.43 (NE x1/2). The last outputs were blocked, so I will keep every printout short. First I'll check which kernel variables survived and list the architecture document's section headers, so I can plan the weight edit.

```python
import json, os
ks=[k for k in globals().keys() if not k.startswith('_')]
print(len(ks)); print(ks[:150])
eff=json.load(open('/work/effects.json')); print({k:(len(v) if hasattr(v,'__len__') else v) for k,v in eff.items()})
arch=open('/model/ARCHITECTURE.md').read(); print(len(arch))
print([l for l in arch.splitlines() if l.startswith('#')])
```