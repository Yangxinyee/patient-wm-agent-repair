The tensor list confirms the head: `value_bin` is a 192→192→17 MLP shared across all reading codes, with `last_numeric` taking 8 inputs. A memory signal added to the shared input would move MAP predictions as well as glucose ones unless it is gated by code. I'll read the head section of the architecture notes, with drug words masked, to see which inputs depend on the code being predicted.

```python
L_ = txt.splitlines()
for i in range(54, min(len(L_), 80)):
    s = re.sub(r'(?i)insulin', 'D8', L_[i]); s = re.sub(r'(?i)norepinephrine', 'D9', s)
    print(i, s[:170])
```