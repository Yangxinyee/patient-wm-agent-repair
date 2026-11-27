The last output was a short per-slot summary. Its context was lost in the restart, so I can't read it reliably. The repeated filter refusals suggest that long or very detailed outputs are the problem. From here I will keep outputs short and refer to the series by code number: code 8 and code 9 are the two infusions, code 6 is the finger reading and code 7 is the arterial reading.

Next I'll list the helper functions still in the kernel and re-read the first part of the architecture file. I need the exact state update before I build a dedicated memory channel. If helpers like `stay_bundles` and `enc_bundle` are still there, I'll reuse them. If they are missing, I'll rebuild only what the edit needs.

```python
import types
fns=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print(len(fns)); print(' '.join(fns)[:1200])
A=open('/model/ARCHITECTURE.md').read(); print(len(A))
print(A[:2500])
```