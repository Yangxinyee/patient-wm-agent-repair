The config gives 4 attention heads (48 dimensions each). The FFN uses exact GELU, which my notes confirm.

Now I'll build the prelude on CPU at the 2,319 drug bundles. The 7 tokens are:
- the previous bundle's h, decayed by the time flow, plus core.slots (4 tokens);
- e_k and u_k from my exact replicas;
- z_in(z), with z the original posterior mean.

From the attention, I'll print per head how much weight the 4 slots put on each non-slot token (e_k, u_k, z). Weight near 0 on z would explain why the z route is dead, and also tells me which carrier can work. If the u_k token gets substantial weight, a carrier through u_k's value is viable.

```python
import torch, math
fa_name = [n for n in NAMES if 'flow' in n]; print(fa_name, [SHAPES[n] for n in fa_name])
st = A['stay_id']; hr = A['hour']
rows = np.where(dr)[0]
prev = rows - 1
ok = (prev >= 0) & (st[np.maximum(prev,0)] == st[rows])
print('drug rows with prev in stay', ok.sum(), 'of', len(rows))
```