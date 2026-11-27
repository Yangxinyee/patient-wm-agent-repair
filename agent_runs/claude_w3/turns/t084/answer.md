The last check showed that no local reimplementation exists in the kernel yet. There are four late helpers (`lev`, `cost`, `prev_h`, `anchor_fit`) whose purpose I don't remember, and torch is not imported.

My conclusion is that I need a faithful local copy of one model step, the encoders, the prelude, the blocks and the coda, to design and check a dose-memory edit precisely. Next I will import torch and print the model config plus the names and shapes of the prelude and block tensors. These fix the head count, the norm placement and the attention structure. If the config gives the number of heads and the tensor names follow the coda's pattern, a reimplementation is feasible.

```python
import torch, json
print(VOCAB['config'])
print(' | '.join(f"{n_[11:]}:{tuple(SHAPES[n_])}" for n_ in NAMES if n_.startswith('core.prelude.0')))
print(' | '.join(f"{n_[12:]}:{tuple(SHAPES[n_])}" for n_ in NAMES if n_.startswith('core.blocks.0.0')))
print(' | '.join(f"{n_}:{tuple(SHAPES[n_])}" for n_ in NAMES if n_.startswith(('core.age','gap','z_post','z_prior','z_in','encode.','embed.cat','embed.code','embed.kind','embed.ekind','embed.kq','embed.text'))))
```