Nothing has been run yet, so there are no results to interpret. First I need to learn how the model computes, what the records contain, and whether my notebook already exists (in case this is a restart). I will read the architecture file, the data README and any existing notes. If the notes are empty, this is a fresh start. If they are not, I should continue from them.

```python
import os, json, textwrap
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print("=====NOTES=====")
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else "(no notes yet)")
print(os.listdir('/work'))
```