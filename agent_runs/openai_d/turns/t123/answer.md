The cached correction uses a gated insulin-innovation decoder, so its strength can be changed analytically without retraining. I’ll measure the local factual cost across several strengths before installing another version. A stronger setting with sufficient NLL margin would justify a paired GPU test; otherwise I’ll keep the current candidate.

```python
import numpy as np
import torch
import dis

print("=== LOCAL AMPLITUDE SWEEP ===")
for strength110 in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print("strength", strength110, "result", local104(strength110))

print("=== CURRENT CONSTRUCTION ===")
print("replacement neurons:", pair104)
print("removed-neuron mean activations:", hbar104)
print("causal slope:", theta104)
print("tilt:", tilt104)
print("decoder step:", decoder_step104, "gate strength:", gate_strength104)
print("innovation std / active std:",
      float(np.std(R104)), float(np.std(R104[:, active104])))
print("fit104:", fit104)
print("gate104:", gate104)
print("fit result keys:", {k: list(v.keys()) if isinstance(v, dict)
                         else type(v).__name__ for k, v in fits99.items()})

print("=== CONSTRUCTOR CALCULATION ===")
dis.dis(local104)

with open("/work/notes.md", "a") as f:
    f.write("""
## Runs 108–110: causal and construction recovery
Immediate residual-dose effects are stable across hour/dose strata:
insulin approximately -1.99 per rate unit; NE approximately +3.29.
Cumulative targets use controlled impulse kernels and remain provisional.
Recovered exact local amplitude evaluator and existing gated construction;
testing scaling analytically before another weight installation.
""")
```