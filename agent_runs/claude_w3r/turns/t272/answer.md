No channel is built yet: impl4.py only lists the names of the kernel functions, with no source. I will build the channel now in four stages:

1. Head-2 free value dims carry the drug rates into the slot tokens.
2. A rank-1 term in `core.b_in` writes them into h.
3. The y-direction `a_y` carries them to the heads.
4. Gated dead-unit pairs in `value_bin` turn them into a mean shift.

First I need the exact tensor names and shapes for these parts, plus the kernel arrays I'll reuse. If the prelude attention uses split q/k/v/out weights, stage 1 is a simple row and column edit.

```python
keys=['prelude','b_in','c_out','value_bin','embed_transition','out_norm','coda','z_head','query']
for n in NAMES:
    if any(k in n for k in keys): print(n, SHAPES[n])
for v in ['U_act','NEG','pa','a_y','dy_','DWC','SX','UX','GR','GRa','inF','ONw','ONb']:
    x=globals().get(v); print(v, type(x).__name__, getattr(x,'shape',None))
print('Wt keys sample', [k for k in Wt.keys()][:5], len(Wt))
```