This is the first step, so there are no results yet. Before planning anything I need to know how the model computes, what the records contain, and whether a notebook from an earlier session already exists. I'll read the architecture document, the data README and any existing notes, and take a quick look at the vocabulary and tensor names. If `/work/notes.md` already exists, this is a restart and I should pick up where I left off; if not, I start fresh.

```python
import os, json
print(os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-6000:])
print(open('/model/ARCHITECTURE.md').read())
```