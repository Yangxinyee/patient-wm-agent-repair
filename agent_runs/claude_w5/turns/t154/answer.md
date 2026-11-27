Bundles are `(time_h, [atom dicts])`. The non-core list was cut off at `filter_gate`, which I hadn't noted before and which may matter for the head. So I'll print the rest of the list, and one insulin bundle so I know the atom fields (rate, dose, kind).

```python
rest = zn[[i for i,n in enumerate(zn) if n.startswith('filter_gate')][0]:]
print("remaining:", len(rest))
print("; ".join(f"{n}{tuple(SHAPES[n])}" for n in rest)[:1600])
cnt_ = 0
for t_b, ats_b in b_:
    if any(a.get('code')=='insulin' for a in ats_b):
        print("bundle t=", t_b, ats_b)
        cnt_ += 1
        if cnt_ >= 2: break
```