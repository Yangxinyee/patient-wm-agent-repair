The tensor list gives the core's layout:
- **Prelude:** one layer (norm1 → self-attention with fused kv and no biases → norm3 → FFN of width 768) over the 7 tokens.
- **Recurrent block:** adds cross-attention to the memory M (norm2).
- **Coda:** another slot transformer layer.

The prelude's self-attention is the only way the u_k and z tokens can reach the slots. Before choosing a carrier, I want the attention weight from the slots to the u_k and z tokens at drug bundles, computed on CPU from replicated inputs.

My earlier replicas of ctx, e_k, u_k and z_post should still be in the kernel. I'll list the kernel's functions and main arrays so I can reuse them rather than rebuild.

```python
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_')}
fn = sorted(k for k, t in g.items() if t == 'function')
arr = sorted(k for k, t in g.items() if t in ('ndarray', 'Tensor', 'DataFrame', 'dict'))
print('FUNCS', fn)
print('ARRS', arr[:120])
```