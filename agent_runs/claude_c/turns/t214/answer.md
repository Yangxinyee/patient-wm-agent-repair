The head-0 ×4 factual cost is spread across all reading types and time bins. It includes +0.006 to +0.016 at readings more than 12 h after any drug record. So head 0 also carries reading information, and scaling it is non-selective whatever the gain. The drug-specific modules are `encode_action.*` (it only touches drug bundles) and the u_k input to `z_post`. Before choosing the next lever, I'll list the exact tensors in those modules and in the prelude layer, with shapes and no values.

```python
for n_ in NAMES:
    if any(s in n_ for s in ['prelude','encode_action','z_post','z_in','numeric']): print(n_, SHAPES[n_])
```