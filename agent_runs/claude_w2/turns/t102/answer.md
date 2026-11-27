ARCHITECTURE.md is short and was printed in full. It adds several details my notes lacked:
- The readout is the predicted mean (sum of p(b) × bin centre), not the drawn value, so it moves directly with the value-head distribution.
- The core keeps a memory M of past measurement-bundle vectors e only. Dose information can therefore persist only through h.
- The recurrent block has self-attention over slots, cross-attention to M with age keys, and an FFN.
- The update is `h <- a_bar*h + softplus(delta_in)*b_in(E) + (u-(h+E))`, then `coda(c_out(h))`.

My plan for the edit is a correction in the value head proportional to the state-decoded deviation of recent doses from the sliding-scale policy. That needs exact local copies of the core step and the head. Then I can make unlimited states (including under counterfactual doses) to train decoders and run local rollouts. Next I print the tensor names and shapes for core, encode, encode_action, heads and norms so I can code the step precisely. If the core tensors match the described layers, I can implement the step and verify it against the fwd_3 states next.

```python
import re
for nmx in NAMES:
    if nmx.startswith(('core','encode','attr_heads.value_bin','query_norm','z_head','out_norm','gap','type_emb','embed_transition','last','filter','embed.code')):
        print(nmx, SHAPES[nmx])
print([k for k in VOCAB.keys()], VOCAB.get('config'))
```