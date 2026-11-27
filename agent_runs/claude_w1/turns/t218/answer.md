The ridge decoders barely respond in the replica's edit states. The lag-1 insulin decoder moves 0.26 for a dose change of 1.6, and it leaks -0.16 from the NE x2 edit. So the state carries little cleanly decodable dose information along these directions. A head edit built on decoded state is either weak or noisy.

The head input also contains a last-rate-bin embedding, which may give a direct, noise-free dose signal. I will check in ARCHITECTURE.md exactly what the value head conditions on.

```python
A=open('/model/ARCHITECTURE.md').read()
import re
for ln in A.splitlines():
    l=ln.lower()
    if ('last' in l and ('rate' in l or 'dose' in l or 'status' in l or 'numeric' in l)) or 'with_code' in l:
        print(ln.strip()[:220])
```