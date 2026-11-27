That was my own bug: an earlier run reused the name `re` for an array, which shadowed the regex module. I'll re-import it under another name and run the same search of ARCHITECTURE.md, to see how u is formed when a bundle has no action atoms.

```python
import re as rx
for kw in ['empty','no action','encode_action','actions']:
    for m in list(rx.finditer(kw,A_))[:2]: print(kw,'|',A_[max(0,m.start()-150):m.start()+250].replace('\n',' ')); print('..')
```