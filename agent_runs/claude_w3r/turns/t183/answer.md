The last result (run 170) checked a linear probe that reads the insulin and norepinephrine rates. It holds in the factual run and in all four dose-edit conditions, including the out-of-range norepinephrine ×2 case: slope about 1, bias under 0.03, small residual spread. So a clean dose signal is available inside the model. My notebook does not record runs 118 to 170, so first I need to recover what was built: the earlier true-effect estimates, the impl files and the kernel variables. Then I can continue the plan of a dose channel plus head readout. Success here means I find the probe and channel code and the effect targets. If I don't find them, I must rebuild them from the files.

```python
import os
txt=open('/work/notes.md').read()
print(len(txt))
idx=txt.find('1 slope by band')
print(txt[:idx][-6000:])
for f in sorted(os.listdir('/work')):
    p='/work/'+f
    print(f, os.path.getsize(p) if os.path.isfile(p) else sorted(os.listdir(p)))
from collections import defaultdict
gg=defaultdict(list)
for k,v in list(globals().items()):
    if not k.startswith('_'): gg[type(v).__name__].append(k)
for t,ks in gg.items(): print(t, len(ks), ' '.join(sorted(ks))[:1500])
```