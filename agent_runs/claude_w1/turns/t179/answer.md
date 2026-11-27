The last run compared my local copy of the output head with the tool's factual means on two stays. They agree to within about 2 mg/dL or mmHg, which matches what my notebook already says about this head replica. The conversation then restarted. I only have the last 8,000 characters of my notebook and no edits are saved yet. Before I build the head-level edit, I need two things back: the earlier notebook sections and the list of helper functions still alive in the kernel. The earlier sections hold how I estimated the true effects (−14.17 / +7.09 / +4.82 / −2.41 on setA) and the original model's dose_edit readouts. I will only read my own notebook and kernel state, using no files outside the allowed set. This step tests no hypothesis; it restores context so I don't redo work.

```python
txt = open('/work/notes.md').read()
print(len(txt))
print(txt[:max(0, len(txt)-8000)][:14000])
print('=====GLOBALS=====')
import types
g = [k for k,v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType)]
print(len(g)); print(sorted(g))
```