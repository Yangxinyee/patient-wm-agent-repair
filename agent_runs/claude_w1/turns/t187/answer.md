The z prior gives a mean and a log-sd from ctx, and evaluation averages probabilities over 4 prior draws. To simulate draws in my replica I need the prior's tensor names and how z feeds the head and the state update. I'll print the surrounding section of ARCHITECTURE.md, with lines truncated.

```python
for i in range(34, 62): print(i, lines[i][:240])
print([n for n in NAMES if 'z_prior' in n or 'z_head' in n])
```