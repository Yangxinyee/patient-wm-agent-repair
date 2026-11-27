I have recovered the targets from my notes. The training records give these true effects:
- Insulin lowers finger glucose with an impulse response of about -2.0, -2.4, -2.1, -1.7, ... per unit/h, about -12.3 for a sustained unit/h. It is linear and the same across subgroups.
- Norepinephrine raises MAP with a response of about +3.35, +2.6, +1.75, ... per mcg/kg/min, about +9.6 sustained.

On the 100 tuning stays that implies:

| Readout | Truth | Original model |
|---|---|---|
| insulin x2 | -13.8 | -2.15 |
| insulin x1/2 | +6.9 | +1.23 |
| norepinephrine x2 | +4.6 | +0.34 |
| norepinephrine x1/2 | -2.3 | -0.06 |

So both effects are 6 to 14 times too weak. The model's one-step dose sensitivity is -0.35 versus a true -2.0 for insulin, and +0.11 versus +3.35 for norepinephrine.

Every amplification so far has failed: input embeddings, the action encoder output (layer-normalised, so it saturates), and z-writers (buried). A value-head probe is too costly in factual fit, because the dose signal in q is weak and noisy.

Writing the dose into the state in a way the value head can read needs exact architecture knowledge, which the kernel restart wiped. The tensor list was also cut off mid-way. This run prints the rest of the tensor names and the architecture sections on the state update, view, heads and action path. I am looking for any pathway from a drug bundle to the next reading's value head that bypasses the core.

- **If** the per-code last-value feature (last_numeric) is built from a multi-code input, a cross-code term from insulin's last rate into glucose queries would be a clean, cheap dose channel.
- **If not**, I fall back to a cost-limited value-head readout. That works mainly for norepinephrine, whose signal in `out` at MAP readings is strong.

```python
import re
i0 = NAMES.index('core.prelude.0.ffn.0.bias') if 'core.prelude.0.ffn.0.bias' in NAMES else 70
print('NAMES tail:', ' | '.join(f"{n}{tuple(SHAPES[n])}" for n in NAMES[i0:]))
arch = open('/model/ARCHITECTURE.md').read()
print('ARCH len', len(arch))
parts = re.split(r'\n(?=#)', arch)
print('HEADERS:', [p.strip().split('\n')[0][:50] for p in parts])
keys = ['core','state','view','head','action','latent','update','slot','prelude','step','bundle','time','last','numeric','query','z ']
out=[]
for p in parts:
    h = p.strip().split('\n')[0].lower()
    if any(k in h for k in keys):
        t = re.sub(r'[ \t]+',' ', p); t = re.sub(r'\n\s*\n+','\n', t)
        out.append(t)
s='\n'.join(out)
print(s[:6400])
```