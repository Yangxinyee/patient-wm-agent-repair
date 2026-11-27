The kernel has a `step` function plus several checking functions (`step_err`, `step_err2`, `step_err3`, `zpost_check`, `zcheck3`). So I evidently implemented the recurrent core and was checking it against the saved states. Next I'll print the source of `step` and `step_err3` to see what they compute. Then I'll re-run the latest check to learn how closely my local core matches the GPU states. Close agreement (relative error around 1e-3) means I can design the edit locally and use few forward passes. A large error means the reimplementation still has a bug to find.

```python
src = inspect.getsource(step)
print(len(src)); print(src[:3500])
```