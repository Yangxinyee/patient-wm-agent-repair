Before doing anything else I need to see how the model computes, what the records contain, and whether an earlier session left notes. So my first step only reads the architecture file, the data README and any existing notebook. A notebook would mean the conversation was restarted and I should pick up from there. If none exists, I start fresh and create one.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=================== DATA README")
print(open('/data/README.md').read())
print("=================== NOTES")
print(open('/work/notes.md').read() if os.path.exists('/work/notes.md') else "NO NOTES YET")
print(os.listdir('/work'))
```