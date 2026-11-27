I have now re-read the whole step. The only way a drug record reaches the state is through the prelude attention from the slot tokens to the u_k token (or to the z token). The value head reads only the state plus the same-code "last" register. So any dose effect must pass through the state.

Next I will look at the core tensors: shapes, how close `c_out` is to the identity, and the decay parameters. These decide whether a single h-direction can act as a controllable leaky memory. If `c_out` is near-identity and the decays are per-dimension, a dedicated memory dimension is feasible. If `c_out` mixes everything, I will switch to a readout of the existing state.

```python
with open('/work/notes.md','a') as f:
    f.write("\n## Session 6\n- Re-read ARCH. Door for dose info into state = prelude attention only (u_k / z tokens). Head reads out, h-after-flow, z-prior, own-code register.\n")
core=[n for n in NAMES if n.startswith('core.')]
print(len(core)); print([(n.replace('core.',''),tuple(SHAPES[n])) for n in core])
```