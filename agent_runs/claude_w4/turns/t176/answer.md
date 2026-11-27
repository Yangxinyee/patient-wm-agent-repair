I've just restarted. Run 155 regressed the new slot state on reading surprises. It found one direction per signal, the same across all four slots (cos 1.0). Its partial R2 is tiny (glucose 0.015, MAP 0.047) next to a typical update norm of about 1000. That suggests surprise information is a very small part of the state update. Runs 144–155 are not in my notebook, so first I'll re-read the architecture file, the earlier part of the notebook and the kernel variables. Then I can decide how to finish the injection edit (plan v2: carry the dose surprise into the core state so later glucose and MAP predictions move).

```python
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch)); print(arch)
n = open('/work/notes.md').read()
print('NOTES LEN', len(n))
print(n[:max(0,len(n)-8000)][:7000])
print(sorted([k for k in globals().keys() if not k.startswith('_')]))
```