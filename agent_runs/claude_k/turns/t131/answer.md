Sending end atoms through the measurement encoder (into e) is right. The error over all 400 action bundles drops from 0.055 to 0.0054 with bare end atoms. End-only bundles are still off at 0.044, so the end atom's embedding probably lacks one categorical term.

I'll repeat the categorical-row search under the new routing on the end-only bundles. I'll also test the end atom carrying its dose with no presence flag. If a variant reaches about 0.001, the action-bundle posterior is fully exact.

```python
best = sorted((err_route('none', en), en) for en, _ in extras)
print([(round(e, 4), en) for e, en in best[:6]])
ex2 = [(a, b) for a in ('action1', 'status1', 'status0', 'action0', 'route1') for b in ('unit3', 'rate_unit2', 'unit5', 'rate_unit3', 'none')]
def err_route2(exs, js=endonly):
    tot = sum((dict(extras)[x] for x in exs if x != 'none'), np.zeros(192))
    extras.append(('tmp', tot)); r = err_route('none', 'tmp', js); extras.pop(); return r
print(sorted([(round(err_route2(p), 4), p) for p in ex2])[:4])
```