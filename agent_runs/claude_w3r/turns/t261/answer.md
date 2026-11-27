The grep closes the latent-z idea. At evaluation the `z` in the view is drawn from the prior, so editing the posterior cannot carry the dose into the reading head, and no prelude head attends the `z_in` token. The only route is design (alpha): prelude head 2 (which attends the action token) → slot tokens → E → `b_in` → h → out → view → reading head.

Building that efficiently needs the validated helper functions. Their source is missing from the kernel because they were exec'd. The files /work/impl*.py probably hold that source. Next I list the `def` lines in those files to see which file defines what and with which arguments. If they are there, I can reuse the validated local core step and prelude-attention code instead of rewriting it.

```python
import os
for fnm in ['impl.py','impl2.py','impl3.py','impl4.py']:
    s=open('/work/'+fnm).read()
    print('==',fnm,len(s))
    for l in s.splitlines():
        if l.startswith('def '): print('  ',l[:150])
```