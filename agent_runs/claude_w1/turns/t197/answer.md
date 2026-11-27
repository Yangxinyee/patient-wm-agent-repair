The run failed because multiprocessing could not pickle the worker. Kernel functions aren't attributes of the real `__main__` module. Registering the worker on `sys.modules['__main__']` lets pickling find it by name, and fork gives the children the kernel's state. I'll rerun the same build; success means I get factual rows and edit responses for most of the 1,500 stays within the time guard.

```python
import sys
sys.modules['__main__'].proc_stay = proc_stay
t0=time.time(); GEN=[]
with ctx_mp.Pool(8) as pool:
    for res_ in pool.imap_unordered(proc_stay, stays_all, chunksize=4):
        GEN.append(res_)
        if time.time()-t0>780: pool.terminate(); break
print('stays done', len(GEN), 'time', round(time.time()-t0,1))
pickle.dump(GEN, open('/work/gen_main.pkl','wb'))
print('rows', sum(len(g_['code']) for g_ in GEN), 'edits', collections.Counter(k_ for g_ in GEN for k_ in g_['ED']).most_common())
```