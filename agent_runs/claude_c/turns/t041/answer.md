The last forward pass shows that my two rank-1 edits on `embed.numeric.weight` moved the model only a little. Insulin x2 went from -1.85 to -2.51 and insulin x1/2 from +0.78 to +1.66. Norepinephrine went from negative for both edits to +0.14 and -0.02. My estimated true effects on these stay-hours are -13.1, +6.5, +4.75 and -2.4, so the edit is far too weak, and too weak for norepinephrine in particular.

The conversation was restarted, so I no longer remember how the two edits were built or how the model computes. Next I will re-read the architecture, the full notebook, the saved operations and vectors, and the live kernel variables, then plan a stronger and better-targeted edit. This run only gathers information. It confirms the plan if the docs show where dose enters the model (the numeric embedding and the value/dose path), so I can choose a tensor whose rescaling gives a roughly linear dose response.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read())
print("=====OPS=====")
print(ops())
print(open('/work/ops.json').read()[:3000])
print("=====VECTORS=====")
print(os.listdir('/work/vectors'))
print("=====NOTES (full length)=====")
n = open('/work/notes.md').read()
print(len(n))
print(n[:6000])
print("=====GLOBALS=====")
g = [k for k in globals().keys() if not k.startswith('_')]
print(g)
```