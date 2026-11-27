The tool rejects edit v1 outright: NLL rises from 1.519 to 18.3. Even though the gates were clean on the replica, the large gate and decoder weights blow up in the real model. The likely cause is z: the tool's head input includes z_head(z) at sampled z (four prior draws), while my replica used the prior mean. Any high-gain direction that overlaps the z subspace gets amplified. Other inexact parts of the head input may add to this. I need the exact composition of the head input and the structure of z_head, so that gate and decoders can be made orthogonal to the z contribution. I will print the relevant architecture lines and the tensor shapes.

```python
import re
arch=open('/model/ARCHITECTURE.md').read().split('\n')
idx=[i for i,l in enumerate(arch) if 'with_code' in l or 'z_head' in l or 'view =' in l]
shown=set()
for i in idx[:4]:
    for j in range(max(0,i-2),min(len(arch),i+4)):
        if j not in shown: print(j,'|',arch[j][:160]); shown.add(j)
print([ (n,SHAPES[n]) for n in NAMES if 'z_head' in n or 'embed.code' in n or 'last_numeric' in n][:10])
```