This is the start, so there are no results yet. My first step is to read the model's architecture and the data description, then start the notebook. That tells me how doses enter the model, which tensors could carry a dose effect, and what the records contain. I have no hypothesis yet. The aim of this run is to learn the structure, and nothing in it can confirm or refute anything.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print(os.listdir('/model'), os.listdir('/data'), os.listdir('/work'))
```