Pass 23 (head-0 query rows ×3, i.e. sharper head-0 attention) left the readouts at the original values (−1.83 / +0.75 / −0.51). Attention sharpness is not what limits the dose effect, so I will drop those 48 ops. Before choosing a new lever I need to re-read how the action token reaches the state. I will log the result, clear the ops, and print only the section outline of ARCHITECTURE.md to keep the output short.

```python
clear_ops()
with open('/work/notes.md','a') as f:
    f.write("\n## Restart-8\n- Pass 23: prelude q rows 0-47 x3 (sharpen head0): -1.83/+0.75/-0.51 -> no effect. Cleared ops.\n")
a=open('/model/ARCHITECTURE.md').read()
print(len(a))
for i,l in enumerate(a.split('\n')):
    if l.startswith('#') or 'prelude' in l.lower()[:60]:
        print(i, l[:110])
```