This is the start, so I have no results yet. First I need to see how the model computes and what the records contain. I will also check whether a notebook from an earlier session already exists in /work. Then I will plan the causal analysis. I am not testing a hypothesis yet; this step only gathers information.

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