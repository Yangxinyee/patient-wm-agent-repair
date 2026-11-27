The core update is a diagonal state-space step: `h ← a_bar·h + softplus(delta_in)·b_in(E) + (block increments)`, then `y = coda(c_out(h))`.

A dose can reach the state only through attention. One route is the prelude's attention from slots to the bundle token. The other is the blocks' cross-attention to the memory M of past bundle vectors, which already has age-coded keys.

The cross-attention route looks most promising. It can re-read past doses at every step, and the age encoding gives a built-in time kernel, so no accumulator would need to survive the coda.

To choose between the routes I need the core tensor shapes (head count, biases, what the coda is) and how e_k is built. If the cross-attention has per-head projections with biases and M holds raw e vectors, I can design one dose-reading head. If not, I'll fall back to the z-posterior or prelude route.

```python
cn=[n for n in NAMES if n.startswith('core.') or n.startswith('embed')]
s=' '.join(f"{n.replace('core.','c.')}{tuple(SHAPES[n])}" for n in cn)
print(len(cn)); print(s[:2600])
print('\n'.join(f"{i}:{L[i].strip()[:170]}" for i in range(7,30) if L[i].strip())[:2200])
```