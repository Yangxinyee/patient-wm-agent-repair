I now have the full local engine. The rollout code processes all arms and paths in one batch. With `t_hide=1e9` and `want_nll=True` it becomes teacher-forced, and `store` exposes every head input. So I can run counterfactual teacher forcing: the same recorded readings with scaled doses. From that I can measure how the head input responds to dose changes, which is the basis for a decoder-based head correction.

To define the correction targets I need the exact data layout and timing (hourly dose grid, reading times, policy function). That is in the early part of my notebook, which I haven't seen since the restart. Next I print it.

```python
s=open('/work/notes.md').read()
print(len(s))
print(s[:len(s)-7900])
```