`c_out` is a dense mixing matrix, not near the identity (diagonal about 0, distance to I about 17). The per-step retention a_bar is only about 0.3 to 0.45. So h is rebuilt each step from the prelude output, and a dedicated memory dimension would need many coordinated edits. Next I will list the head tensors, especially `pending_mlp` and `filter_gate`, to see whether any input channel outside the recurrent state carries the running dose. If such a channel exists, it is a much simpler lever.

```python
oth=[n for n in NAMES if not n.startswith('core.') and not n.startswith('embed.') and not n.startswith('encode')]
print(len(oth)); print([(n,tuple(SHAPES[n])) for n in oth])
```