u_k reaches the state only as one token in the prelude transformer layer: tokens [h+slots ×4, e_k, u_k, z_in(z_k)], with only the four slot outputs kept. z enters the same way via `z_in`. That explains the weak z channel: the slots probably give the z token little attention, and pre-norm rescales it. To choose a strong, near-linear entry point, I'll read the full core-step description (lines 36 to 60).

```python
for i in range(30,62): print(i, L[i][:230])
```