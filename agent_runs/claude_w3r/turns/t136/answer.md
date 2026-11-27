The last run tested my local copy of the model: swapping the insulin action token for one encoding a different rate moved the predicted next glucose by only about −0.1 and +0.25 mg/dL on average. The records suggest roughly −2 mg/dL per unit/h at one hour. So the model has almost no direct path from dose to the next glucose reading. This confirms the plan in my notes: I need to add a pathway that carries the dose forward, minus a compensation for the usual dose given that glucose level.

Since this is a restart, I'll first re-read the part of my notebook not shown here: the original model's readouts, the target effects and what is defined in impl.py. Then I'll plan the edit without repeating work. If the notebook already holds the targets and original readouts, I can go straight to building the channel. If not, computing the targets comes next.

```python
import re, os
txt = open('/work/notes.md').read()
print('notes length', len(txt))
head = txt[:max(0, len(txt)-8000)]
print(head[:7000])
print('-----')
for f in ['/work/impl.py', '/work/impl2.py']:
    s = open(f).read()
    print(f, len(s), 'defs:', re.findall(r'^def (\w+)', s, flags=re.M))
g = [k for k, v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ not in ('module',)]
print('globals:', g[:150])
```