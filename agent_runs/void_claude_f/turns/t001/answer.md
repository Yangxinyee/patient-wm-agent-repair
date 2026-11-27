I'm starting fresh. Before changing anything I need to know three things: how the model computes, what the records contain, and whether an earlier session already left a notebook in /work. My first step is to read the architecture file, the data README, the vocabulary summary and any existing notes. None of this tests a hypothesis yet; it is the information I need to form one.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print("=====NOTES=====")
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read())
else:
    print("no notes yet")
print(os.listdir('/work'))
```