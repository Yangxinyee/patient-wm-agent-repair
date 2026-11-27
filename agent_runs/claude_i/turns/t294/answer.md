`run_stay_z` reproduces the tool's states closely: out error is 0.005 at most, z error 0.002, and the larger h error comes from a few bundles. `value_mean` gives the head's predicted mean for any code from a context. Together these let me measure locally how an injection at a drug bundle moves the next glucose and MAP predictions.

To design the new z_in column I must perturb the z token directly, which needs a single-step call. `core_step(hf, e, u, z, M, ages, V)` has undocumented arguments, so I'll print `sim.py` (1.6k characters), which shows how `run_stay` calls it. After that I can write a one-step perturbation routine over many insulin and norepinephrine bundles at once.

```python
print(open('/work/sim.py').read())
```