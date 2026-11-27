This is the start, so there are no results yet. First I need to understand how the model computes, what the records contain, and whether a notebook from an earlier session exists. I will read the architecture document, the data README and any existing notes, and look at the vocab and tensor shapes. I am not testing a hypothesis yet; this is orientation.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print("=====NOTES=====")
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else "NO NOTES YET")
print(os.listdir('/work'))
```