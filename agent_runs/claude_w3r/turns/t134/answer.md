`inspect` cannot recover the source of the kernel's head functions. They were defined from strings in an earlier session, so their signatures are unclear.

`/work/impl2.py` contains a `Head` class that should implement the local value head. Next I will load it in its own namespace and list its methods with their signatures and short bodies. I can then use it to measure how much the predicted glucose and MAP means depend on u.

- **Confirms:** `Head` computes with_code and value-bin probabilities from a state (out, h), the bundle time gap and the register, so I can reuse it directly.
- **Refutes:** it is incomplete. Then I will write a compact head function from ARCHITECTURE.md instead.

```python
ns2={}; exec(open('/work/impl2.py').read(), ns2); Head=ns2['Head']
src2l=src2.splitlines()
st_=[i for i,l in enumerate(src2l) if l.startswith('class Head')][0]
print('\n'.join(src2l[st_:st_+60]))
```