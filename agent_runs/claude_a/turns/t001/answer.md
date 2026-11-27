Nothing has run yet, so there are no results to read. I'll start by getting oriented. First I'll check whether /work/notes.md exists (in case this conversation was restarted). Then I'll read the architecture description and the data README, which tell me how the model computes and what the records contain. This first step tests no hypothesis. Its job is to show me how doses enter the model and what the record columns are, so I can later estimate true drug effects from the records and find where in the weights the dose→reading pathway lives.

```python
import os, json
print(os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-5000:])
print(open('/model/ARCHITECTURE.md').read())
print("=====================README")
print(open('/data/README.md').read())
print(os.listdir('/work'))
```