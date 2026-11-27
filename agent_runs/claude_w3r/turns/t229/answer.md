Restricting the read to the quiet directions does not help enough. With 10 to 90 directions the insulin residual at action bundles stays at 0.35–0.54, and the noise elsewhere grows as I add directions. The rate is not encoded linearly within that quiet subspace of u.

The fix should come earlier: edit the action encoder itself so that u carries the rate exactly, along a quiet direction. Whether that is possible depends on how `encode_action` builds u from the rate, and in particular on whether a linear "size" path exists after the MLP. I will read that part of ARCHITECTURE.md and list the encoder tensors.

```python
arch=open('/model/ARCHITECTURE.md').read()
i0=arch.find('encode_action'); print(arch[max(0,i0-300):i0+1800])
print([ (n,SHAPES[n]) for n in NAMES if n.startswith('encode_action')])
```