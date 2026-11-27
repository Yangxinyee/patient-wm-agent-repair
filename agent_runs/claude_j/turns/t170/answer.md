The state section gives the full update. Time-flow decay comes first, then the prelude yields E, then the recurrent block updates `h <- a_bar*h + softplus(delta_in)*b_in(E) + (u-(h+E))`. Finally `h = coda(c_out(h))` and `out = out_norm(mean_slots(h))`.

A dose memory that lives inside one dimension of h would have to pass through `c_out` and `coda` at every step. Whether that can be done cleanly depends on what those two modules are, for example a near-identity linear map plus a residual MLP.

So next I will print the names and shapes of the `core.*` tensors outside the prelude and blocks, plus the `encode_action.*` and `attr_heads.*` names. This is a short listing.

If `c_out` is square and `coda` is a residual MLP, a dedicated dimension can carry a decaying memory. If not, I will route the signal elsewhere, for example by reading it from the state with a fitted readout.

```python
ks=[n for n in NAMES if n.startswith('core.') and not (n.startswith('core.blocks') or n.startswith('core.prelude.0'))]
print(' '.join(f"{n[5:]}{list(SHAPES[n])}" for n in ks)[:1500])
print(' '.join(f"{n}{list(SHAPES[n])}" for n in NAMES if n.startswith('encode_action') or n.startswith('attr_heads'))[:1200])
```