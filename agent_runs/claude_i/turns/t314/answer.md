The architecture excerpt shows the value head's register inputs (`last_numeric`) are per-code: the last value, rate and age of the same code. So no insulin feature reaches the glucose value head directly.

One possible direct route is the "pending" input p in the filtered view, which may hold the running infusion rates. It seems to be used only for action predictions, though. The other candidates are the memory M and the z_head(z) path.

**This step:** read the core and head sections (lines 40–76) in full, truncated per line. I want to know:
- exactly what feeds the reading-value head;
- what p contains;
- how the prelude, memory and z enter the state.

If p (the record's pending doses) or anything computed from the record reaches reading predictions, that is a strong edit path. If only the view reaches them, I must build the two-stage write-then-read route.

```python
A = open('/model/ARCHITECTURE.md').read().splitlines()
print(' | '.join(f'{j}:{A[j].strip()[:140]}' for j in range(40, 77) if A[j].strip())[:5200])
```