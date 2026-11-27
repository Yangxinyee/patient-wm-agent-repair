The only flag-like global the replica reads is `T`, used by `lin`, `linp` and `mha`, and it's currently the integer 150. The code still runs, so `T` may be used as an attribute name (like `.T`) rather than a global, or it may really be a stale variable. I'll print `lin`'s and `mha`'s instructions around `T`. If the global `T` really is used, its value probably changed after the replica was validated, which would explain the mismatch.

```python
for f in [lin, mha]:
    il=list(dis.get_instructions(f))
    for i_,x_ in enumerate(il):
        if x_.argval=='T':
            print(f.__name__, ' | '.join(f'{y_.opname[:9]}:{str(y_.argval)[:14]}' for y_ in il[max(0,i_-4):i_+3]))
```