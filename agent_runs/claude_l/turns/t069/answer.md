The last run compared my re-implementation of the model's state updates with the states the forward tool returned (forward pass 2). Most stays match exactly (error 0). A few stays differ by up to 0.04 in a handful of dimensions. Each of those bundles holds a `glucose_finger` reading in a high bin (bin 13), so the cause is probably one feature-encoding detail, such as an edge or clipping in how readings are binned, not a structural error.

Since the conversation restarted, I first need to recover my working context: the architecture document and the functions and variables already in the kernel. The next step is then to find that small mismatch and confirm the re-implementation is exact. That is what lets me design a core-level edit that gives the model the multi-hour insulin and NE memory it lacks.

If the mismatched bundles share a feature value near a bin edge or the clipping limit, the hypothesis "a binning detail" is confirmed. If not, I will look for something else, such as event ordering inside a bundle.

```python
import re, json, os
print(open('/work/notes.md').read()[:3000])
print('=====ARCH=====')
arch=open('/model/ARCHITECTURE.md').read()
print(len(arch))
print(arch[:12000])
print('=====GLOBALS=====')
g=[k for k,v in globals().items() if not k.startswith('_') and callable(v) and getattr(v,'__module__',None)=='__main__']
print(g)
print([k for k,v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ in ('ndarray','DataFrame','dict','list','Tensor')][:200])
```