The kernel still has many variables, and /work has a numpy encoder and head (enc.py, head.py). fwd_2 (factual, original model) and fwd_3 (states, original model) were never written up. Before planning the edit I need the architecture back in my head, so I'll re-read ARCHITECTURE.md in short chunks. This is orientation, not a test.

```python
s = open('/model/ARCHITECTURE.md').read()
print(len(s))
print(s[:3500])
```