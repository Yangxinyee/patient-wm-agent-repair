This is the start, so there are no results yet. My plan:

1. Read the architecture description and the data README.
2. Check whether `/work/notes.md` already exists, in case this is a restarted conversation.
3. Estimate the true drug effects from the training records. Doses follow the patient's condition, so a naive regression would be confounded. I need a design that uses variation in dose unrelated to the patient's state, for example rate changes driven by protocol or by chance, adjusted for prior readings.
4. Find where in the network the dose acts, and edit those weights so the model's dose response matches the estimate without changing its factual predictions.

This first step only gathers information, so it cannot yet confirm or refute anything. What I'm looking for is how doses enter the model (input embedding or projection weights) and what the records contain.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====README=====")
print(open('/data/README.md').read())
print("=====NOTES EXIST?=====", os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[-5000:])
print(os.listdir('/work'))
```