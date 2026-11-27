The full local trajectories on train stays drift from the tool by about 2e-4 relative at every bundle. On the tuning stay, though, the first drug bundle alone produced a 2e-3 difference from matching inputs. My notes mention single-step errors of up to 6e-3 on some bundles. That pattern looks like a specific atom feature being computed differently at some drug bundles, rather than float32 noise. A likely suspect is the rate/dose bin assignment when a value lies exactly on a bin edge (rates come in 0.5 steps), where 'left' and 'right' conventions differ.

My edit decodes dose information from exactly these drug-atom-derived features, so it would be hit hardest. To test this I'll look at the first drug bundle of stay 2020000014 and at what `avec`/`enc2` call for binning. Then I'll redo that single step from the tool's previous state:
- If the error is around 1e-6, the step itself matches and the difference comes from elsewhere.
- If it is around 2e-3, the drug atom encoding differs and I'll try bin-side variants.

```python
B14 = TB[2020000014]
print(B14[3])
for f in ['avec','enc2','senc','fine_bin','fbin','fb_','fbv']:
    fn = globals()[f]; co = fn.__code__
    print(f, co.co_varnames[:co.co_argcount], '| names:', co.co_names, '| consts:', [c for c in co.co_consts if not isinstance(c, type(co))][:40])
print({d['code']+'/'+d['attribute']: d['fine_bin_edges'] for d in VOCAB['numeric'] if d['code'] in ('insulin','norepinephrine')})
```