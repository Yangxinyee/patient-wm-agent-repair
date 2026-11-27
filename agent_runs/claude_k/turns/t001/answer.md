This is the start of the task, so there are no results yet. First I'll check whether a notebook already exists in /work/notes.md, in case this is a restarted conversation. Then I'll read the architecture description and the data README. I need to know how the model computes, where dose information enters, and how the records are laid out before I can plan. Two things would show I'm not at the start: an existing notebook, or existing edits. If I see either, I'll resume from the notes and not start over.

```python
import os, json
print("notes exists:", os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-6000:])
print("edited:", edited())
print(open('/model/ARCHITECTURE.md').read())
```