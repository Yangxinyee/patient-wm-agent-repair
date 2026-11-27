The head functions are not finished. `head_probs` and `cmp_head` take unknowns (type index, status index, LN layout), so the local value head was still being matched against fwd_3. The leftover numbers from run 106 (about 6.8 mg/dL off on glucose_finger) suggest it does not match yet.

Before finishing the head, I need to settle the design of the edit, because that decides which head details matter. The key question is where a persistent dose memory can live. Doses enter only through the action encoder, then the prelude, then h. The head reads h fairly directly via out_norm(mean over slots of the flowed h).

Next I will re-read the Input and State sections (time flow, decay, input gate, pending features) and print the config values. That shows which per-dimension decay and write parameters exist for building a leaky dose integrator in h.

```python
print({k: cfg[k] for k in cfg})
print('\n'.join(lines[7:54])[:3600])
```