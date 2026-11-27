The last run checked whether a correction could act through the head. A view shift of norm 0.095 moves both glucose predictions by 1 unit and leaves MAP unchanged. For norepinephrine the matching shift is 0.243. However, decoding a stored dose residual back out of the state is noisy: the gain per bundle was about 0.1 with a noise sd of about 0.28. So a memory loop that reads back from the state looks unreliable.

My notebook stops at the plan from run 177, so runs 177–254 are not recorded. I have 46 Python runs left. Before building anything I need to see which helper code and variables already exist in the kernel and in /work.

What I will do next depends on what this inventory shows:
- If the helpers (replica step, rollout, z-token injection, decoders) are already there, I will build the simplest workable edit. The z token will copy u, plus a calibrated dose-residual term at drug bundles.
- If they are missing, I will rebuild only the minimum needed.

```python
import re, types
for f in ['inj.py','wm2.py','wm2_rollout.py','tf.py','setup.py']:
    try:
        src=open('/work/'+f).read()
        defs=re.findall(r'^def (\w+)\(', src, flags=re.M)
        print(f, len(src), defs[:30])
    except Exception as ex: print(f, 'ERR', ex)
g=[k for k,v in list(globals().items()) if not k.startswith('_') and not isinstance(v, types.ModuleType)]
arrs=[(k, tuple(globals()[k].shape)) for k in g if hasattr(globals()[k],'shape') and not callable(globals()[k])]
fns=[k for k in g if callable(globals()[k]) and not isinstance(globals()[k], type)]
oth=[k for k in g if k not in fns and k not in dict(arrs)]
print('n globals', len(g))
print('arrays', arrs[:50])
print('funcs', fns[:90])
print('other', oth[:70])
print('edited', edited())
```