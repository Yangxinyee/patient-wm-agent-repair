`view_from` fails the same way, but the error comes from my `np.asarray` wrapper, not the function. It probably returns a tuple (q plus extra parts) of mixed shapes. I need to see its return structure before the SNR analysis.

This run inspects one call: its type, the length and shapes of its parts, and which part matches Q99[0]. Then I can extract q correctly.

```python
o_=view_from(xo[kR[0]],dtR[0])
print(type(o_))
if isinstance(o_,(tuple,list)):
    for i,p_ in enumerate(o_):
        a=np.asarray(p_) if not isinstance(p_,(tuple,list,dict)) else None
        print(i,type(p_).__name__, None if a is None else a.shape)
        if a is not None and a.size>=192:
            print('   diff vs Q99[0] %.3e'%np.abs(a.ravel()[:192]-Q99[0]).max())
elif isinstance(o_,dict): print(o_.keys())
else: print(np.shape(o_))
```