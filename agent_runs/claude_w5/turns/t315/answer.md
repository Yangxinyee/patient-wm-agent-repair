Run 285 localises the artifact. The wrong-sign ×½ responses come almost entirely from stays whose recorded hour-12 insulin rate sits exactly on a bin edge, chiefly r = 1.0.

Recorded rates lie on a 0.5 grid, and the rate bin edges repeat at those grid values. So the left and right bin conventions give very different bins: for r = 1.0, left gives bin 4 and right gives bin 8. Under the left convention used for factual encoding, the in-between bins are never seen in training.

Stays at r = 1.0 give a negative glucose shift under both ×2 and ×½. Stays at r = 3.5 give clean, symmetric, even oversized responses. My leading hypothesis is that the dose_edit rollout re-bins drug atoms from hour 12 on with the right-edge convention in all three runs (recorded, ×2, ×½). My decoders, fitted on left-binned encodings, then misread those edited inputs.

Before testing this per stay locally, I need the calling conventions of my kernel helpers, which I can no longer read as source. This run prints their argument and global-name lists compactly, plus the structure of one bundle. Next run I can then compute local h13 shifts per stay under left, right-everywhere and right-edited-only binning and correlate them with the tool's per-stay h13 shifts. A clearly higher correlation under one convention confirms it.

```python
import inspect
def info(fn):
    f = globals().get(fn)
    if f is None: return f"{fn}: MISSING"
    c = getattr(f, '__code__', None)
    if c is None: return f"{fn}: no code"
    return f"{fn} args={c.co_varnames[:c.co_argcount]} names={tuple(n for n in c.co_names)[:30]}"
for fn in ['avec','fbin','fb_','fb_orig','mk_fb2','enc2','scale_atoms','make_bundles','run_tf','mystep','pred_mean','n3_of','tok_n1','build_edit','last_before','isdrug','one_step_from_tool']:
    print(info(fn)[:420])
selA = json.load(open('/work/sel_tuning_A.json')) if 'json' in globals() else __import__('json').load(open('/work/sel_tuning_A.json'))
print('selA type', type(selA).__name__, (list(selA)[:3] if not isinstance(selA, dict) else list(selA.keys())[:5]))
sid0 = 2020000016
B0_ = make_bundles(tu[tu.stay_id == sid0])
print('n bundles', len(B0_), type(B0_[0]).__name__)
k12 = [i for i, b in enumerate(B0_) if 'hour' in str(type(b)) or True][:0]
for i, b in enumerate(B0_[:60]):
    s = repr(b)
    if '12.08' in s and ('insulin' in s or 'norep' in s):
        print('bundle', i, s[:400]); break
print('WCUR is WE:', globals().get('WCUR') is globals().get('WE'))
try:
    print('b_in match:', float((to64(load('core.b_in.weight')) - WE['core.b_in.weight']).abs().max()))
except Exception as e: print('b_in check fail', repr(e)[:100])
```