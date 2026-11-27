The step details matter. `h` is updated as an SSM-style term, `a_bar*h + softplus(delta_in)*b_in(E)`, plus the residual output of the blocks. Then it is rewritten by `coda(c_out(h))`. A dose channel survives only if `c_out` and `coda` can pass one direction almost unchanged. Also, the heads average over 4 draws of z, which explains part of my head-replica mismatch. Next I'll list the names and shapes of all `core.*` tensors related to `coda`, `c_out`, `b_in`, `delta_in`, `a_in` and `flow`, plus the encoder tensors.

```python
sel_=[n for n in NAMES if any(k in n for k in ['coda','c_out','b_in','delta_in','a_in','flow','out_norm','slots','encode_action','z_post','z_prior','z_in','z_head'])]
print(len(sel_)); print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in sel_)[:1900])
```