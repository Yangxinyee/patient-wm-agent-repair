The architecture notes confirm the update rule: the write-in strength `softplus(delta_in)` is a per-dimension parameter rather than an input-dependent one, and the time decay is per dimension through `flow.a`. Next I'll read how the bundle inputs `e_k` and `u_k` are built, because the dose has to enter through them.

```python
for i, line in enumerate(txt.splitlines()):
    if re.search(r'e_k|u_k|atom|encod|numeric|rate|dose|bundle', line) and i < 70:
        print(i, line[:170])
```