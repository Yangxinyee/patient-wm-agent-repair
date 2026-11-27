I'm starting fresh, so I'll get oriented before forming any hypothesis. I need three things: how the model computes (ARCHITECTURE.md), what the records contain (README.md), and whether /work/notes.md already exists from an earlier session. Once I've read these I'll know which tensors carry the dose information. Then I can plan how to estimate the true drug effects from the records, separately from the confounded associations.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("="*100)
print(open('/data/README.md').read())
print("="*100)
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read())
```