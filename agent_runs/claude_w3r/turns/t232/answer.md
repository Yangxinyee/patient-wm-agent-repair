Two findings. First, z_rate is linear: z = (rate − median)/IQR, clipped to ±10. For insulin that is (rate − 1.5)/1.0, for norepinephrine (rate − 0.46)/0.6, and the dose is a fixed multiple of the rate. Second, my earlier search found no dead units in the encoder MLPs.

A dead-unit gate in the encoder is therefore out, but a purely linear edit is possible. mlp2's output layer (`encode_action.mlp2.2.weight`) is linear in its 768 hidden activations, so a rank-1 term δ⊗φ with φ·hidden ≈ rate would inject the rate into u along a chosen direction δ. I would aim δ at a direction that is quiet for both the other tokens and natural u tokens. Head 2 (value edit, then out-projection) would carry it onward with little noise.

First I need to rebuild the encoder exactly. I will inspect the code objects of `atom_vec`, `u_of` and `mlp_t` to see their arguments and the attributes they use.

```python
for nm in ['atom_vec','u_of','mlp_t','LNt','enc_eu']:
    if nm in globals(): info(globals()[nm])
print(globals()['atom_vec'].__code__.co_varnames[:20])
print(globals()['u_of'].__code__.co_varnames[:20])
```