On the sel100 tuning rows, my estimates of the true readouts (with standard errors) set the targets:

| Edit | Target | Model now |
|---|---|---|
| insulin x2 | −14.1 (±1.2) | −0.39 |
| insulin x1/2 | +7.1 (±0.6) | +0.15 |
| norepinephrine x2 | +3.8 (±0.27) | −0.66 |
| norepinephrine x1/2 | −1.9 (±0.13) | −0.24 |

So the model badly underestimates the insulin effect and gets the norepinephrine sign wrong.

Next I will save these targets to the notebook. Then I will look at how the action encoder exposes dose magnitude, which is where a dedicated dose signal would have to come from. A clear dose-proportional direction in u_k would confirm the injection plan.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- TARGETS on sel100 tuning (refined theta): ins_x2 -14.1 (se1.2), ins_half +7.1 (.6), ne_x2 +3.8 (.27), ne_half -1.9 (.13). Model orig: -0.39,+0.15,-0.66,-0.24. Edit names: insulin_x2, insulin_half, ne_x2, ne_half.\n")
print('ok')
```