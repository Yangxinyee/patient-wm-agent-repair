The 300-stay set has 13,036 bundles: 3,388 with insulin, 2,292 with norepinephrine, 892 with both. `prelude_parts` returns three arrays for the 7 prelude tokens: two of width 192 and the per-head values (4×48). I don't know which of the first two is the post-attention residual. To fit the write gate I need to compute that residual and its pre-FFN norm exactly, and to add FFN units. So I'll read the source of `prelude` and `mh` in lib2.py and write a version that exposes these internals.

- **Can proceed:** the code shows the standard pre-norm order (attention on norm1, add, FFN on norm2, add, prelude_norm of the slot tokens). I can then insert units directly.

```python
L=open('/work/lib2.py').read().splitlines()
for i,l in enumerate(L[:31]): print(i+1, l[:160])
```