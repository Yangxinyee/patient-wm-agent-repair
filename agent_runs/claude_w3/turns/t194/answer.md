The first half shows the state update. h is a 4 x 192 slot state with a per-dimension exponential decay `exp(-dt*exp(core.flow.a))`, timescales from 10 minutes to 2 years. Drug atoms enter only through the action encoder u_k, at the prelude. So a dedicated h-dimension with a tuned decay could hold a leaky dose memory. Next I will read the second half, the rest of the step, the heads and how a dose edit is read. That shows how h is written (b_in / c_out) and what the value head consumes. The design is confirmed if the head reads `out` (or h) linearly enough that one added channel can shift the predicted glucose mean.

```python
print(arch[2700:])
```