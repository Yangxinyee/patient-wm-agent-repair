The helpers are still in the kernel, including `scale_head`. Op 3 shows alpha 1.0, which means head0 x2, not the x2.5 the notebook describes. So the current state may differ from what was measured in pass 26. I'll check the alphas of all 48 head ops and read `scale_head`'s source, so I know exactly which configuration is loaded before spending a pass. Confirmation would be a single alpha value across ops 3-50.

```python
import inspect
print(sorted(set(round(x['alpha'],4) for x in o[2:])), sorted(set(x['side'] for x in o[2:])))
print(inspect.getsource(scale_head)[:1200])
```