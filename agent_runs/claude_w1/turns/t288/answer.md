On factual rows, the decoders vI and vN cost far more than they gain: one decoder unit is about 1 dose unit at lag 1, yet their robust spread is about 10 and 2.4 dose units. I will drop them. The ridge decoders in DEC had a decode error of about 0.5 (insulin) and 0.2 (norepinephrine) in earlier checks. Those are usable, so I will build the value-head edit on them. Next I need the exact tensor names of the value head, the layout of DEC, and whether a normalisation sits before the head's first layer. That decides whether a decoder fitted on the head input can be wired straight into a hidden unit.

```python
import json, re
hn=[n for n in NAMES if ('value' in n or 'transition' in n or n.startswith('head'))]
print(len(hn)); print([(n,SHAPES[n]) for n in hn][:24])
print('DEC keys',list(DEC.keys()))
for k,v in DEC.items():
    if isinstance(v,dict): print(k,{kk:getattr(vv,'shape',type(vv).__name__) for kk,vv in v.items()})
    else: print(k,type(v).__name__,getattr(v,'shape',None))
arch=open('/model/ARCHITECTURE.md').read().split('\n')
sel=[l for l in arch if re.search('value_bin|transition|with_code|head',l)]
for l in sel[:14]: print('|',l[:170])
```