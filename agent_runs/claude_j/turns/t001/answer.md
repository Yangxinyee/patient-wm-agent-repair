This is the start, so there are no results yet. First I need to understand how the model computes and what the records contain. I will read the architecture description, the data README and any existing notebook, in case this conversation was restarted. If there is already a notebook, I continue from it rather than starting over.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print("=====NOTES=====")
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else "NO NOTES YET")
print(os.listdir('/work'))
```