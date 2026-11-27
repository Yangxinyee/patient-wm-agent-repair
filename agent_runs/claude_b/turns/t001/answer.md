This is the start, so there are no results yet. First I will read the model's architecture description and the data README, and set up my notebook. I need to know how the model represents doses and readings before I can work out which weights carry the dose-to-reading effect. Nothing is being tested yet. This step only gathers what I need to form hypotheses.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=================== DATA README ===================")
print(open('/data/README.md').read())
os.makedirs('/work', exist_ok=True)
if not os.path.exists('/work/notes.md'):
    with open('/work/notes.md','w') as f:
        f.write("# Notebook\n\n## Run 1\n- Read ARCHITECTURE.md and data README.\n")
print(os.listdir('/model'), os.listdir('/data'), os.listdir('/work'))
```