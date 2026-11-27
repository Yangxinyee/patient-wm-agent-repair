The kernel still has its helpers, including `stay_bundles`, `enc_bundle`, `step`, `prelude_attn`, `zpost_err` and `mha`. The architecture file confirms the setup. The state has 4 slots of h, a memory of past bundle vectors, and `out`. Each step applies a per-dimension time decay to h, then a latent z, then the prelude. The prelude is a pre-norm layer over the tokens [h + slots, e_k, u_k, z_in(z)].

Next I'll print the rest of the file. I need the exact path from the prelude slot outputs into h, and the path from h to `out` and the value head. Those decide whether a dedicated h channel can carry a code-8/code-9 memory that the head reads.

```python
print(A[2500:5234])
```