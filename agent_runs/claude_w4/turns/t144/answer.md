Heads 1–3 differ in how they treat non-drug bundles, so before designing the injection I want an exact handle on the rest of the step. I'll aim for a CPU replica of the whole step: recurrent blocks with cross-attention to memory, plus coda. Checked against the GPU states, it would let me compute how any prelude injection changes the post-bundle state, and confirm gates at every bundle type, without spending forward passes.

The missing piece is the memory M and its age embedding. I'll list the tensor names for age/memory and the remaining core tensors.

```python
for n in NAMES:
    if n.startswith('core.') and not any(k in n for k in ['prelude','blocks.0.0.','coda','b_in','c_out','delta_in','a_in','out_norm','slots','flow']):
        print(n, SHAPES[n])
print([l[:160] for l in lines if 'memory' in l.lower() or 'age' in l.lower()][:8])
```