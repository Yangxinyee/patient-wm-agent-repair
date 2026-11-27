The rescan settles it. Dims 2 and 67 are expensive on the drug-heavy sel2 stays (+0.0026 and +0.0039), which is why the old build failed. About a dozen dims cost at most 0.0003 on both sets: 79, 99, 50, 49, 104, 150, 153, 70, 64, 191, 149, 137. I'll move the memory design onto those.

The truth kernel for insulin is hump-shaped and lasts about 9 hours (peak at lag 2, then roughly 0.72× per hour). A single exponential memory dim with a half-life of about 2–3 hours should approximate it once the model's own carry-forward is accounted for.

To reuse the existing writer `build_ins2` with new dims, I need its arguments. `dvec`, `d0` and `keep` were set earlier, and I overwrote a global named `keep` in run 248. So I'll print just `build_ins2` and `reenc`, compressed onto a few lines, plus whether `dvec` exists. If `keep` is easy to recompute (for example, a selection of head-3 value dims), I'll rebuild with the cheap dims next. Otherwise I'll write a fresh, simpler writer.

```python
src=open('/work/lib4.py').read()
i0=src.find('def build_ins2'); blk=src[i0:].splitlines()
print(' ;; '.join(l.strip() for l in blk if l.strip())[:2600])
i1=src.find('def reenc'); print('REENC:', ' ;; '.join(l.strip() for l in src[i1:i0 if i0>i1 else None].splitlines()[:8])[:500])
print('dvec' in globals(), getattr(globals().get('dvec'),'shape',None), 'd0', globals().get('d0'))
```