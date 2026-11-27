The kernel still has the replica tools: `step`, `run_variant*`, `head_mean*`, `q_for`, `view_parts`, and atom builders. It also holds the kernels `KI`/`KN` and the truth table `R`. ARCHITECTURE.md is short (5.2k chars).

To design a clean dose channel I need the exact state-update equations. The key question is whether some path carries a signal from the drug atom to the value head nearly linearly, for example a residual path through the coda. I will print only the "State and one step per bundle" section, lines 30–53, which is about 2k chars.

If that section shows residual or identity paths, a hand-built dose-memory channel is feasible. If every path is mixed through layer norms and MLPs, I will fall back to a ridge readout from the head query.

```python
L_=txt.splitlines()
print('\n'.join(L_[30:54])[:2600])
```