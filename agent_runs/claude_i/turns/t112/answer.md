The architecture shows how the drug inputs reach the predictions:
- They enter as u_k, a vector from encode_action computed from the rate and dose embeddings.
- u_k feeds the z posterior and the prelude transformer, which shapes the state h.
- The value head then reads `out` and a "view" built from h.

So the dose edit works only through the rate and dose features: their z-scores, their bins and the numeric embedding. An edit to encode_action or the embedding could change how strongly a dose moves the state. But that is hard to control through the transformer.

An alternative is to make the value head's glucose prediction depend more strongly on recent insulin through some feature the head can read. The head sees no drug history directly; it sees only the view. The view contains `out`, which summarizes the state after encoding the drug bundles.

Before choosing, I need my existing helper code: enc.py and head.py, and what is in the kernel. Next I will list my kernel globals and print the two helper files. If the files already hold a full numpy re-implementation of the step and head, I can simulate edits offline. If they do not, I will extend them.

```python
import types
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and k not in ('In','Out','exit','quit','get_ipython')}
print(len(g))
print(g)
for f in ['/work/enc.py', '/work/head.py']:
    s = open(f).read()
    print('=====', f, len(s))
    print(s[:6000])
```