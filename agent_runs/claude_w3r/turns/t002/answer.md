The notebook is empty and nothing has been run yet, so I am starting from scratch. First I will read the architecture description and the data README, and start the notebook. This step tests no hypothesis. It gives me what I need to plan the analysis: how the model computes, the data columns, and where the dose inputs enter.

```python
import os, json
os.makedirs('/work', exist_ok=True)
with open('/work/notes.md', 'a') as f:
    f.write("# Notes\n\n## Session 1 (restart 1)\n- Step 1: reading ARCHITECTURE.md and data README.\n")
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
```