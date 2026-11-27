The probe edits are now dropped, so the model is back to the original. Most kernel variables are left over from replica debugging. The data-analysis objects, such as the policy model and residuals, seem to have been lost when the kernel died at run 93.

Before requesting a factual pass, I need to know which helpers still exist. `/work/setup.py` and `/work/wm.py` are short, so I'll print just their function and section headers.

```python
for fn in ['/work/setup.py','/work/wm.py']:
    src = open(fn).read()
    print('=====', fn, len(src))
    for line in src.splitlines():
        if line.startswith('def ') or line.startswith('class ') or line.startswith('#') or ('=' in line and not line.startswith(' ') and len(line)<100):
            print(line[:100])
print(open('/work/batches.json').read()[:300])
```