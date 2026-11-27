I have lost context in the restart, so first I need to re-read the architecture and check pass 10. The notebook says pass 10 tested scaling `z_in` by 4 along the glucose-surprise and MAP-surprise directions, but the result was never written down. The two current ops on `z_in.weight` are that test.

This run prints:
- `ARCHITECTURE.md`, to recall the mechanics;
- the contents of `fwd_10`, to see whether the z-token route has any potency on glucose and MAP predictions;
- `ops.json`.

If z-token edits shifted the factual predictions a lot, the z route is a usable lever. If they did nothing, I need another pathway, for example the output heads or the transition embeddings.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print('-----')
for f in sorted(os.listdir('/work/fwd_10')):
    p=os.path.join('/work/fwd_10',f); print(f, os.path.getsize(p))
for f in sorted(os.listdir('/work/fwd_10')):
    p=os.path.join('/work/fwd_10',f)
    if f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
        print('== ',f); print(open(p).read()[:3000])
print(open('/work/ops.json').read()[:2000])
```