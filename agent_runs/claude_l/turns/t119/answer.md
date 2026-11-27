The error regression is noisy (100 stays) but points one way.

- **Insulin, finger glucose:** the model's error moves with the dose residuals (lag 1 −1.36, lag 2 −1.14, lag 3 −0.62, se about 0.7). The policy-predicted parts have mixed signs summing to about +0.45. So the model is roughly right along the normal dosing pattern; it has absorbed the predictable dose into how it reads glucose and under-reacts to the residual.
- **Conclusion:** simply amplifying the dose pathway would add a systematic error proportional to the predicted dose and damage factual fit.
- **What would be right:** add a correction Σ c_j·ε_{t−j}, with ε measured against expected dosing and computed on the rollout's own history. In a rollout this gives correct interventional effects.
- **Norepinephrine:** too noisy to read here.

To implement that correction I need a channel in the core that can hold a decaying dose memory. The update `a_bar*h + softplus(delta_in)*b_in(E)` is a diagonal linear recurrence, but `c_out` and `coda` act after it at every step. I'll inspect their shapes, and whether they are near-identity or residual.

```python
core=[n for n in NAMES if n.startswith('core.')]
print(len(core))
print('; '.join('%s%s'%(n.replace('core.',''),tuple(SHAPES[n])) for n in core))
```