`step_at` returns (y, out) only. `step_parts(k, du, dE)` returns 5 values, including E, but it reads the stay from the globals `rows0`, `B0`, `EU0`. So I'll point those globals at the current stay, call `step_parts` on one drug bundle, and match its outputs against `step_at` to work out which element is E. Success means one output matches y or out exactly and a 4×192 output is a separate, normalised E.

```python
B0, rows0, EU0 = B_, rows_, EU_
r5 = step_parts(ks[0])
print([ (type(x).__name__, tuple(getattr(x,'shape',()))) for x in r5])
for i, x in enumerate(r5):
    x = torch.as_tensor(x)
    for nm, ref in [('y', r1[0]), ('o', r1[1])]:
        if tuple(x.shape)==tuple(ref.shape): print(i, 'vs', nm, float((x.double()-ref.double()).abs().max()))
    if x.ndim==2: print(i, 'row norms', x.norm(dim=1).numpy().round(2)[:4], 'mean', float(x.mean()), 'std', float(x.std()))
```