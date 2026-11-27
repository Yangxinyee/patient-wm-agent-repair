Run 281 shows the cause of the failure. The v4 gate units were moderate (row norms 25–33), and in my replica they separate the event types cleanly, with margins of about 26 and spreads under 1. The two "decoder" units, however, had row norms of about 310 and biases near −190. The original head's rows have norms around 1.35.

Reduced precision alone does not flip those gates: bf16 errors are about 0.5 against margins of 26 or more. So the more likely failure is the per-bundle latent z. The tool draws z from the prior four times, while my replica fixed z at its prior mean. A decoder row with norm 300 multiplies even small z-driven changes in the head input into tens of units, which switched units on even for lab and cuff readings.

Conclusions:
- The gate idea is fine at moderate scale. LayerNorm gammas are in [0.70, 1.51], so the 1/gamma vector is small.
- Reading the dose out of the head's input needs a huge-norm decoder because the dose signal there is weak. That route is not robust and should be abandoned or kept tiny.

Before designing a replacement, I need a pathway where the drug-rate signal is clean and does not depend on z. I'll search the architecture file for how the head input is assembled:
- the register, last value, last rate and status features;
- view, query, prior and posterior;
- the drug token and prelude.

If any of the deterministic head features carries the drug rate, that is the robust route. I'll also list my helper functions so I can reuse them rather than rebuild.

```python
import re, types
txt=open('/model/ARCHITECTURE.md').read().splitlines()
print('lines', len(txt))
kw=re.compile(r'(register|last_|rate|status|view|with_code|query|prior|posterior|prelude|u token|\bu\b)',re.I)
hits=[(i,l) for i,l in enumerate(txt) if kw.search(l)]
print('hits', len(hits))
for i,l in hits[:90]: print(i, l.strip()[:140])
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print('FUNCS', ' '.join(sorted(fn))[:900])
```