The channel had almost no effect on the rollouts: drug A ×2 went from −2.65 to −2.57, and drug B ×2 from +0.75 to +0.79. So the dose signal injected through the action token does not reach the predictions. That fits my earlier replica findings: amplifying the change in u reduced the response, and scaling the numeric dose columns did nothing.

The remaining candidate carrier is the drug event's own bundle embedding, for example a rate‑bin embedding. I will drop the ineffective edit and look up how drug rates enter the bundle embedding (tensor names, shapes, the relevant lines of the architecture notes).
- **Confirmed** if a per‑code rate‑bin embedding exists for the drug codes. Its rate‑dependent part could then be amplified.
- **Refuted** if no such table exists.

```python
import re
drop_edit(KV); drop_edit(BI); print('edited',edited())
print([(n,tuple(SHAPES[n])) for n in NAMES if n.startswith('embed')][:20])
arch=open('/model/ARCHITECTURE.md').read().split('\n')
hits=[i for i,l in enumerate(arch) if re.search(r'value_bin|numeric|bundle|action|rate|bin',l,re.I)]
print('arch lines',len(arch),'hits',len(hits))
for i in hits[:22]: print(i, san(arch[i])[:110])
print(list(VOCAB.keys())[:20])
```