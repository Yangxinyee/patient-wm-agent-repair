Before touching anything, I'll read the architecture description, the data README, and any existing notebook. If I've been restarted, the notebook holds my earlier findings. Nothing has been run yet, so there are no results to interpret. I'm starting by orienting myself.

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