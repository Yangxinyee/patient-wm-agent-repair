The architecture shows why writer v2 is inert. The z posterior mean reaches the state only as one token, `z_in(z_k)`, in the prelude attention layer. u_k is likewise just one context-free token, and the slot outputs are LayerNormed into E. Measured end-to-end, a 0.15 change in z gives about 0.0002 change in `out`, so a linear z write would need to be about 200 times larger, far outside z's natural range. The next step is to measure how much attention the slots actually pay to the u and z tokens. That decides between amplifying through `z_in` from a dedicated quiet z dimension, editing the value path, or another route. First I'll list the helper functions still in the kernel (names only), to see if my reconstruction pieces (`u_rec`, view/prelude rebuilders) survive.

```python
fns=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print(len(fns)); print(' '.join(fns))
print([n for n in NAMES if n.startswith('core.prelude') or n.startswith('z_in')][:30])
```