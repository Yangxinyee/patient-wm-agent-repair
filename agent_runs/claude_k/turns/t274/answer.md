Found it: `mlpf` and `setenc_my` call a global function `L`, probably a linear-layer helper over the tensor dict `T`. I overwrote `L` with the document's lines in run 244, so restoring `L` will fix the helper chain. Since the source isn't available, I'll disassemble the two helpers compactly to see how they call `L` (argument order, use of `T`). Then I'll redefine it as a weights-from-`T` linear map.

```python
import dis
DOCL=L
for f in ['mlpf','setenc_my']:
    ins=[(i.opname,i.argrepr) for i in dis.get_instructions(globals()[f])]
    print(f, ' | '.join(f'{o[:9]}:{a}' for o,a in ins if a)[:900])
```