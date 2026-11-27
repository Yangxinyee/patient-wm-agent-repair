The step replica gives a clear and sobering picture of the prelude route.

- **Shrinkage.** A vector added to the slots after prelude attention is shrunk about 40-fold by the prelude norm. The new slots then change by only about 0.027 per unit, and `out` by about 0.00013.
- **Direction is not the problem.** The best directions are stable and separable: the glucose and MAP gradients are each consistent across bundles (cos 0.999) and nearly orthogonal to each other (cos 0.09).
- **Size is the problem.** Through `out` alone, a 1 mg/dL shift needs an injection of norm about 23, and 1 mmHg about 34. A realistic insulin kick of about 1.5–2 mg/dL per unit/h would mean norms of 35–45 per unit/h. At high rates that approaches half the slot norm.

That estimate may be incomplete. My readout used only the previous `out`, but the context the prediction heads read also includes a layer norm of the time-decayed previous slots. Those change about 200 times more than `out` per unit of injection, so that term could be the dominant channel, and the true requirement could be much smaller.

To settle this I need to know exactly what the prediction heads read: the context alone, z, or both, and how. Next I'll re-read the first part of /model/ARCHITECTURE.md, the authoritative description, rather than infer further. Then I'll compute the full context-based gradient.

```python
txt=open('/model/ARCHITECTURE.md').read()
print(len(txt),'chars'); print(txt[:9000])
```