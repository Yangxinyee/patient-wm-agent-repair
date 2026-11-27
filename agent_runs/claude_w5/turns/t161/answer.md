`head_P` returns a bare integer, so the earlier head helpers are unusable as they stand. I'll rebuild the head from the architecture description and check it against the fwd_3 predicted means. For that I'll re-read the head section of ARCHITECTURE.md, lines 50–80, each line truncated.

```python
for j in range(50, 81):
    if j < len(txt): print(j, txt[j][:230])
```