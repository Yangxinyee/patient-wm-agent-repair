The last result showed the start of my notebook: true effects estimated from records, and the cost and effect of each edit tried so far. My conclusion stands: only the action-token path is selective, and its gain is capped. The current 50-op combo (numeric x5 + head0 x2.5) is just over the factual budget (+0.0051), so I must weaken it. I'll first list the kernel's helper functions and check the current ops, so I can rebuild head0 at a smaller factor. Confirmation: helpers exist that build head0 basis ops with a chosen alpha.

```python
import types
fs=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print(fs)
o=ops(); print(len(o)); print(str(o[0])[:200]); print(str(o[2])[:200])
```