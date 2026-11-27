The `pending`/`filter` path only feeds action prediction: `filter_norm(view + gate*pending_mlp(p))`. It does not touch the value head, so my belief stands that the only dose route to glucose and MAP predictions is u → prelude attention → E → h. I also noted that at evaluation z is drawn from the prior 4 times and the probabilities are averaged; my `head_value` replica already does this.

Next I want the exact wiring surface for the circuit:
- what my `kill()` in lib1.py zeroes (it is short, so I will print it);
- the shapes of the core and head tensors.

With these I can write the insulin circuit:
- memory dim with abar=1 and a chosen flow decay;
- staging dim fed from the prelude head-3 out-projection;
- gated write through two GELU units in the block-0 FFN;
- code-gated readout in the value MLP.

If `kill` already zeroes every reader and writer of a dim, as its 4-dim cost of about +0.001 suggests, I can build on it directly.

```python
print(open('/work/lib1.py').read())
sel=[n for n in NAMES if n.startswith(('core.','attr_heads.value_bin','query_norm','last_numeric','embed.code','embed_transition.value_bin','type_emb','z_head','gap.','last_status','embed.bins'))]
print(' '.join(f"{n.replace('core.','c.')}:{'x'.join(map(str,SHAPES[n]))}" for n in sel))
```