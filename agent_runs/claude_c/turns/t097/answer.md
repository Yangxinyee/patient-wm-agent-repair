Run 85 showed that VOCAB's numeric entries are per-code normalisation tables, not a description of the 8 register inputs. So that path is unconfirmed, and I am setting it aside.

The best-defined remaining idea is the z-space remap, which turns the dose signal into the reading-surprise direction g. Two things have to hold for it to work:
- the z token must actually influence the state at reading bundles;
- g must exist in z-space.

First I'll check what I already built: the candidate g vectors, the selective dose direction, and which forward outputs I have on disk.

```python
import os, numpy as np
for k in sorted(os.listdir('/work')):
    if k.startswith('fwd_'): print(k, os.listdir('/work/'+k))
for nm in ['g','gvec','gsrc','gsrc_b','v','p','e26']:
    if nm in globals():
        x = globals()[nm]; print(nm, type(x).__name__, getattr(x,'shape',None) if hasattr(x,'shape') else (len(x) if hasattr(x,'__len__') else x))
```