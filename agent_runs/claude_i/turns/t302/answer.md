The key lines are there:
- The recurrent block computes `u = blocks(h + E)` and then `h <- a_bar*h + softplus(delta_in)*b_in(E) + (u - (h+E))`.
- The value head adds per-code "last" features: `last_numeric([has_last, log(1+age)/10, z_last_value, z_last_rate])`, plus last value-bin and rate-bin embeddings and a last-status embedding.

I need the exact wording of the core-step and head sections (lines 30–75). Two questions matter:
1. whether the blocks' self-attention reads the memory of past e tokens (which would explain why readings are strong and doses weak);
2. what "last" means for a glucose code, i.e. whether any drug information reaches the glucose head directly.

```python
for i in range(28, 76): print(i, lines[i][:240])
```