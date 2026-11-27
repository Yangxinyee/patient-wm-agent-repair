The atom counts line up, so bundle alignment is fine. Dropping the time flow lowers the error from 0.59 to 0.41, but neither variant is close. So either the encoders as called with default options are off, or the context needs a detail I'm missing.

My old helpers had option switches that probably record what I had already worked out. I'll print the option keys and strings used inside `enc_bundle`, `zparts` and `zpost_err`. If I can identify the validated settings, I can reuse them instead of re-deriving everything.

```python
for fn in ['enc_bundle','zparts','zpost_err','zpost_check2','atom_vec','set_enc_v','HN','G_']:
    c = globals()[fn].__code__
    print(fn, [x for x in c.co_consts if isinstance(x, str) and len(x) < 30][:30])
```