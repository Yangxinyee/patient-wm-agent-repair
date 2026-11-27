The architecture notes confirm that drug atoms reach the state only through the action token u_k. Measurement heads see dose information only through the core state. The rate‑bin table `embed.bins.rate` feeds u_k, not the measurement heads, so amplifying it would again run through the path that failed.

Before choosing between the remaining options (a different injection point in the core step, or a reduced head edit), I need the exact step equations, to see whether u_k has a more direct, persistent route into the state.
- **Confirmed (new route):** u_k enters the step additively somewhere other than the prelude attention.
- **Refuted:** the prelude attention is the only entry point.

```python
for i in range(28,60):
    l=arch[i].strip()
    if l: print(i, san(l)[:150])
```