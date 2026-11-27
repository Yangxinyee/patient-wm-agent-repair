The stale prelude op is gone and the op list is empty. The architecture confirms the dose signal reaches the state mainly through the action token u_k in the prelude attention. I already showed the z route has no effect, and the reading heads never see drug rates. Next I'll list the tensor names and shapes for `encode_action`, the prelude and the embedding. Then I can see whether the attention has separate q/k/v projections. A value-projection edit along the action-token direction would be selective, and a `encode_action`-only edit is selective by construction.

```python
for n in NAMES:
    if n.startswith('encode_action') or n.startswith('core.prelude') or n.startswith('embed') or n.startswith('core.z_in') or n.startswith('core.slots'):
        print(n, SHAPES[n])
```