This is the start, so there are no results yet. First I need to learn how the model computes and how the records are laid out. I also need to check whether a notebook already exists in /work, in case this conversation is a restart. I will read /model/ARCHITECTURE.md, /data/README.md, any existing /work/notes.md, and the vocab summary. If notes already exist, I will continue from them rather than start over.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=================== DATA README")
print(open('/data/README.md').read())
print("=================== NOTES?")
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read())
else:
    print("no notes yet")
print(os.listdir('/work'))
```