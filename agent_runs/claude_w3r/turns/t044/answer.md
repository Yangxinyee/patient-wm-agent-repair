The last output listed my kernel variables and my saved re-implementation, /work/impl.py. The file only covers the input side: atom embedding, the set encoders, the context and z_post. So the large errors in run 38 most likely came from comparing my z posterior mean with the states' z (median error 0.4 against a typical |z| of 0.58). The input or z_post step therefore still has a bug.

Before choosing between finishing a full re-implementation and a simpler, head-centred edit, I need the exact module structure. That means the tensor names and shapes of the encoders, z_post, gap, prelude, blocks, coda and heads. I will print them compactly.

What would settle it:
- If z_post or gap has a structure different from the plain Linear-GELU-Linear I assumed (extra layers or norms), that explains the mismatch.
- If the structure matches my code, the bug is in the inputs, for example dt units or END-atom attributes.

```python
import collections
grp = collections.OrderedDict()
for n in NAMES:
    grp.setdefault(n.split('.')[0] if not n.startswith('core.') else '.'.join(n.split('.')[:2]), []).append(n)
for g_, ns in grp.items():
    print(g_, '|', '; '.join(f"{n[len(g_)+1:] if n.startswith(g_+'.') else n}{list(SHAPES[n])}" for n in ns))
```