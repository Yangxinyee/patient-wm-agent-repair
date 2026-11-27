Run 137's inventory came through. The kernel still has the verified local core (`cstep`, `enc2`, `tf_check`), several head-reconstruction helpers (`head_mean`, `head_P`, `eval_full`) and the g-estimates (`gam_ins`, `gam_ne`). The responses after that were withheld, so I'll keep outputs short and strictly technical.

What I know about the model: dose information reaches the core state only through the action encoding `u`. `u` enters `z_post` and the prelude as a token, so the prelude attention from the slots to `u` (and to the `z` token) is the gate any dose-memory edit must pass through. Before designing an edit I need to know which tensors make up the prelude and the other core read/write paths. If slots already attend substantially to `u` in some head, a dose signal can be written there almost linearly. If not, I'll need another write route.

```python
pre = [n for n in NAMES if 'prelude' in n]
print("prelude tensors:", len(pre))
for n in pre: print(" ", n, tuple(SHAPES[n]))
oth = [n for n in NAMES if n.startswith('core.') and 'prelude' not in n and 'block' not in n]
print("other core (non-block):", len(oth))
print("; ".join(f"{n[5:]}{tuple(SHAPES[n])}" for n in oth))
blk = sorted(set('.'.join(n.split('.')[:3]) for n in NAMES if 'block' in n))
print("block groups:", blk[:12])
print("block0 tensors:", "; ".join(f"{n.split('.',3)[-1]}{tuple(SHAPES[n])}" for n in NAMES if n.startswith(blk[0]+'.'))[:900])
```