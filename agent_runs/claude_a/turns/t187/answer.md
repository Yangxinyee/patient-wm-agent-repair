Combined kills cost more than the single-dim costs add up to:

| cheapest dims killed | cost, selection set E | cost, fresh set F |
|---|---|---|
| 4 | +0.0023 | +0.0004 |
| 7 | +0.0055 | +0.0036 |
| 10 | about +0.0105 | about +0.0105 |

So I must use as few dims as possible: about 4 dims fits (about 0.001), 7 does not.

I can cut the design to 3 dims for insulin and 2 more for NE, 5 in all:
- **Insulin:** memory jI, reference jR, raw staging jWI.
- **NE:** jWN and jN, sharing jR.
- **Write:** the prelude attention puts the rate into jWI, which reaches the blocks through E. Two gated GELU units in the first block's FFN read the jWI−jR difference and write straight into the memory dim. That removes the separate gated staging dim.
- **Gate:** for that, the "insulin present" signal must be linearly separable, with margin, from the block FFN input.
- **Read:** one value-MLP unit, gated to glucose_finger through killed dims, whose with_code is a fixed function of code and last bin.

To fit these pieces I need prelude and block internals per bundle. My notes say `tf_loop(P, ds, keep)` stores internals (the stored dict `ST2` has keys aw/E/hf/hn/z). I'll inspect its outputs on the 20-stay set before writing my own prelude function to validate against.

```python
r=tf_loop(PO, ds, keep=None)
print(type(r).__name__, len(r) if isinstance(r,(tuple,list)) else '')
for i,x in enumerate(r if isinstance(r,(tuple,list)) else [r]):
    if torch.is_tensor(x): print(i,'T',tuple(x.shape))
    elif isinstance(x,dict): print(i,'dict',{k:(tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k,v in list(x.items())[:8]})
    elif isinstance(x,(list,tuple)): print(i,type(x).__name__,len(x), type(x[0]).__name__ if len(x) else '')
    else: print(i,type(x).__name__)
print({k:(tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k,v in ds.items()})
print({k:tuple(v.shape) for k,v in ST2.items()})
```