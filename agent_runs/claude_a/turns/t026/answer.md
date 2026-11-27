Run 23 shows the model's one-step dose sensitivities are far too weak in teacher-forced mode.

| | model, randomised within-tier dose | actual records |
|---|---|---|
| insulin, next glucose (lag 1) | −0.01 ± 0.28 | −1.98 ± 0.57 |
| norepinephrine, next MAP (lag 1) | +0.83 ± 0.34 | +3.82 ± 0.81 |

So the model has essentially not learned the drugs' direct effects. Its longer-lag coefficients mostly come from the observed intermediate readings.

The doses are sequentially randomised given the recorded history. So if the model's one-step conditionals had the right dose dependence, the rollout (a Monte-Carlo g-formula with fixed doses) should give the right interventional answers. The fix is therefore to add the missing dose dependence to the one-step predictions.

To decide where to add it, I'll inspect the tensor names and shapes. I want to see whether the value head is an MLP (so a code-gated hidden unit could carry a dose correction) and how the action embedding and encoder are laid out.

```python
for n in NAMES:
    print(n, SHAPES[n])
```