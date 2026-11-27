The local hour-13 check is symmetric, while the tool's is not:

| Hour 13 | Local (teacher-forced to 12:05) | Tool rollout |
|---|---|---|
| insulin x2 | −2.66 | −3.08 |
| insulin x1/2 | **+1.28** | **+0.09** |
| norepinephrine x2 | +0.88 | +1.01 |
| norepinephrine x1/2 | −0.40 | −0.36 |
| edited − original, factual: glucose | +0.05 | +0.66 |
| edited − original, factual: MAP | −0.09 | −1.43 |

The tool's core matches mine exactly when teacher-forced. So the tool's state at hour 13 is not the plain teacher-forced state. Something in the rollout before or at hour 13 differs from my assumption: perhaps which readings are hidden, how drawn values are fed back, or how z is drawn.

The documentation section "How a dose edit is read" is the authority, so I'll print it in full rather than guess.

```python
L_ = open('/model/ARCHITECTURE.md').read().splitlines()
print('\n'.join(f'{i}: {l}' for i, l in enumerate(L_) if 50 <= i <= 95))
```