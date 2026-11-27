Run 69 failed because this kernel is not IPython, so there is no input history (`In`) to recover my earlier code. The note did reach the notebook, since that line ran first.

Without that history, the reliable path is to rebuild the forward pass from scratch, slowly and checked piece by piece. I'll compare each stage against the states from forward pass 2: z posterior mean, h slots, then out. That way the cause of the 8.7% mismatch shows up instead of staying hidden.

First I need the exact tensor names and shapes, the config, and the structure of what is already in the kernel: `Wt`, the format of `bundles_B`, and the states file. I'll print these compactly. If the shapes match the description in ARCHITECTURE.md, I can write the rebuild directly. Anything unexpected, such as extra norm or gate tensors, would point to the earlier mismatch.

```python
import json, inspect
cfg = VOCAB.get('config', {})
print('config:', cfg)
print('vocab keys:', list(VOCAB.keys()))
line = []
for n_ in NAMES:
    line.append(f"{n_}{tuple(SHAPES[n_])}")
print(' '.join(line)[:7000])
print()
print('Wt type/len:', type(Wt), len(Wt))
for fn in ['lin','mlp','lnp','gapf','enc','encA','emb_visit2','make_bundles']:
    if fn in globals():
        try:
            print(fn, inspect.signature(globals()[fn]), globals()[fn].__code__.co_varnames[:12])
        except Exception as e_:
            print(fn, 'err', e_)
```