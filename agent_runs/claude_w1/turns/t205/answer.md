The last result was not delivered, so I learned nothing new from it. The notebook gives the plan and the target effects (-14.17 / +7.09 / +4.82 / -2.41). It does not say how those targets were estimated. Before building the head-level edit I need two things. First, the earlier part of the notebook, to recover how the true effects were estimated and how much to trust them. Second, a check that the key kernel objects still exist (replica functions, decoders, setA, tool residuals). I will keep the printed output short. If the earlier notes show the targets came from a sound causal analysis of the records (for example, regression on randomised dose innovations), I will go ahead with calibration. If not, I will re-estimate them first.

```python
import os, json
txt = open('/work/notes.md').read()
print(len(txt))
print(txt[:7000])
```