This is the start, so nothing has been run yet. Before deciding anything I need to know how the model computes, which tensors control how doses affect the readings, and what the records contain. I'll check whether a notebook already exists in case this is a restart, then read the architecture file, the data README and the vocab.

Nothing is being tested yet. I'm just gathering the facts that later hypotheses will rest on.

```python
import os, json
print(os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-5000:])
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
```