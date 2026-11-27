The function sources cannot be retrieved: they were defined directly in the kernel, so they are not in a file. The notes still recover the key facts about the NE part of `gated3`:

- It uses head units 131 and 159.
- It reads the NE-rate deviation (lag-1 slope +3.27 MAP per unit) and only acts on map_arterial rows.
- At a_ne=3 the tool measured a factual cost of +.0053 per map_arterial row on random stays, about .0013 per reading overall.

To judge whether a combined edit (insulin gain plus a smaller NE gain) can fit the budget, I need three things:
- the call signatures of my helpers (`eval_edit`, `head_run`, `gated3`, `ins_only`, the cost helpers);
- the size of the value head;
- exactly which entries `gated3(a,a_ne)` changes compared with `ins_only(a)`. Then I can build a "combo" edit that adds only units 131/159 on top of the current insulin edit.

If the difference is confined to rows/columns 131 and 159 (plus possibly the TR table), a clean combination is possible. Otherwise I would need to rebuild it.

```python
import inspect
for fn in ['eval_edit','head_run','gated3','ins_only','make_edit','stay_cost','head_nll','nll_of','truth_ne','truth_ins','shift_stats','evalg','head_eval','pair_out']:
    if fn in globals():
        try: print(fn, inspect.signature(globals()[fn]))
        except Exception as e: print(fn, 'sig?', e)
for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']:
    print(n, SHAPES[n])
print('KMAP', KMAP if 'KMAP' in globals() else None)
gi = ins_only(-2.7); gg = gated3(-2.7, 2.0)
print({k: (type(v).__name__, getattr(v,'shape',None)) for k,v in gi.items()}, list(gg.keys()))
for k in gg:
    a = np.asarray(gg[k]); b = np.asarray(gi[k]) if k in gi else None
    if b is None: print(k, 'only in gated3', a.shape); continue
    d = np.abs(a-b); print(k, a.shape, 'maxdiff', d.max())
    if d.ndim==2:
        print('  rows differing', np.where(d.max(1)>1e-7)[0][:20], 'cols differing', np.where(d.max(0)>1e-7)[0][:20])
    elif d.ndim==1: print('  idx differing', np.where(d>1e-7)[0][:20])
# compare saved edit with ins_only(-2.7)
for k,n in KMAP.items():
    if k in gi: print(k, n, 'saved==ins_only(-2.7):', np.allclose(load(n).numpy() if hasattr(load(n),'numpy') else load(n), np.asarray(gi[k])))
```