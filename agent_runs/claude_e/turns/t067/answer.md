The last result shows that the original model has almost no sensitivity to insulin within a tier. Its one-step predicted glucose responds to the dose deviation with coefficients near zero or slightly positive: lag 1 is +0.19 ± 0.26. The records show a true lag-1 effect of −2.0 per u/h. This confirms the earlier dose_edit result, where the model underestimated the insulin effect by about 8x.

My function definitions were lost in the restart, so I need the architecture details again before I can design a head-level edit. Next I will re-read /model/ARCHITECTURE.md and the full notebook. If the value head and dynamics are as I noted (out, 4 slots of h, z posterior), I can plan an edit that leaves the factual states mostly unchanged.

```python
print(open('/model/ARCHITECTURE.md').read())
print("=====NOTES=====")
print(open('/work/notes.md').read()[:6000])
```