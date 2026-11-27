Restart 9 orientation. Where things stand: E4 fixes insulin (x2 −13.93 vs −13.99, x1/2 +7.03 vs +6.99) and NE x1/2 (−2.15 vs −2.43). NE x2 is still weak (+1.03 vs +4.85). My working explanation is that doubled NE doses go outside the recorded range (rate bin 16, large z), so the encoder does not represent them as "more NE".

I do not know what runs 237–238 tested; the "u-path / e-path" errors of about 0.1 suggest a reproduction that did not match. Before planning the NE x2 fix I need three things back:
- the architecture text for the action and state-update path,
- the effect targets,
- the helper functions I wrote.

```python
import json, re
A = open('/model/ARCHITECTURE.md').read()
print('ARCH len', len(A))
print([l for l in A.split('\n') if l.startswith('#')])
print(open('/work/effects.json').read()[:1200])
N = open('/work/notes.md').read()
print('NOTES len', len(N))
for l in N.split('\n'):
    if re.search(r'(?i)true effect|target|TRUTH|beta_|effects.json|linear', l): print('  >', l[:220])
fns = sorted(k for k,v in list(globals().items()) if callable(v) and getattr(v,'__module__',None)=='__main__' and not k.startswith('_'))
print('FNS', fns)
```