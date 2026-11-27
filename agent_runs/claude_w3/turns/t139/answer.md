`step_err3` returns two numbers, (5.255, 0.0614). These look like an absolute error and a 6% relative error. My local core therefore does not yet match the GPU states, and I can't read the old source to find out why.

I'll print the model config (head counts, sizes) and the non-core tensor groups. This decides whether to rebuild the forward pass cleanly from ARCHITECTURE.md plus the config, instead of guessing at the old helpers. If the config gives head counts and dims, a clean rebuild checked bundle by bundle against the saved states is the more reliable route.

```python
print(CFG if len(str(CFG)) < 1500 else str(CFG)[:1500])
grp = defaultdict(int)
for n in NAMES:
    if not n.startswith('core.'): grp['.'.join(n.split('.')[:2])] += 1
print(dict(grp))
```