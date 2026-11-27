Run 171 recovered the notebook. It holds the targets from the records:
- **Insulin:** a structural response of about −2.0, −2.45, −2.1, … mg/dL per 1 u/h deviation. Pooled targets on sel1 are about −13.7 for ×2 and +6.9 for ×½.
- **Norepinephrine:** about +3.45, +2.79, +1.79, … mmHg per 1 mcg/kg/min. Pooled targets about +5.2 for ×2 and −2.6 for ×½.

The plan was a channel from the action token u through the prelude, b_in and h to the value head. Work after run ~117 is not in the notes. The impl files are the best record of it, so next I'll read impl4.py and the function list of impl3.py. If they contain the channel or probe code (e.g. a rank-1 edit or dead-unit readout), I can resume from there. If not, I'll rebuild that step.

```python
for fn_ in ['impl4.py','impl2.py']:
    print('=====',fn_); print(open('/work/'+fn_).read())
s3=open('/work/impl3.py').read()
print('===== impl3 defs')
for i,l in enumerate(s3.split('\n')):
    if l.startswith('def ') or l.startswith('class ') or (l and not l.startswith(' ') and '=' in l and len(l)<120): print(i,l)
```