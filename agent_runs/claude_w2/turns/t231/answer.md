The core injection failed. With κ = 0.1 (an injected vector about 10% of the slot-residual norm per typical dose), factual value NLL got much worse: finger glucose 1.456 → 1.675, overall 1.608 → 1.734. Meanwhile the head's causal decoding barely improved (slope 0.57 vs 0.48 at μ=10). The top singular direction of the linear skip path is one the network relies on heavily, so writing into it does much damage while its nonlinear processing still scrambles the signal. I am dropping this edit; nothing was saved.

Before probing stage by stage where the dose information gets scrambled, I want to rule out a path I may have overlooked. The tensor list includes `pending_mlp` and `filter_gate`/`filter_norm`, which none of my notes or implementations mention. If the architecture encodes currently running infusions (with their rates) for the heads, that would be a direct, editable dose path to the value head, far better than going through the core. I will search ARCHITECTURE.md for those parts and print the relevant passages.

```python
arch=open('/model/ARCHITECTURE.md').read(); print(len(arch))
import re
for kw in ['pending','filter','running','active infusion','last_numeric','register']:
    for m_ in re.finditer(kw,arch):
        s=max(0,m_.start()-300); print('---',kw,'@',m_.start()); print(arch[s:m_.start()+500]); break
```