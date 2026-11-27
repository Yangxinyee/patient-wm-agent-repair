The tail shows the full local toolkit is still present:
- the exact core with injection hooks: cstep_h, cstep_r for injection into E, cstep_K;
- weight-setting helpers: setW/resetW;
- local rollout and head functions: free_run, substep, mhead;
- the run-185 regression results: Xb at the block-1 FFN input, a discriminant w_sc, residual decoders w_t, and block-1 FFN weights W1b/b1b/W2b.

The notebook has nothing on runs 179–185, so I'll rebuild the design in plain steps and record it this time.

Before choosing where the main injection lives, I need the exact stack layout. If block 1 is the last block, a gated unit in its FFN writes into h_new linearly, so I can aim the injection exactly along the belief direction. If there are more blocks after it, I have to correct for their response. Next I print the weight-name structure compactly (prefix → count, block indices, FFN shapes).

```python
from collections import Counter, defaultdict
keys = list(W64.keys())
pref = Counter(k.split('.')[0] for k in keys)
print(dict(pref))
blk = defaultdict(list)
for k in keys:
    p = k.split('.')
    if p[0] in ('blocks','core','prelude','coda') or 'block' in k:
        blk['.'.join(p[:3])].append('.'.join(p[3:]))
for kk, vv in list(blk.items())[:14]:
    print(kk, len(vv), ','.join(vv)[:150])
print([ (k, tuple(W64[k].shape)) for k in keys if 'ffn' in k and ('block' in k) ][:8])
```