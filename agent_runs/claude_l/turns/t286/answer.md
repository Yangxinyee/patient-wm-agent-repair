The previous output was blocked, and my notes stop at run 177. Runs 177–263 left files I have not documented: `inj.py`, `tf.py`, `setup.py`, `s1data.npz`, `jac.npz` and `tf_train.npz`. I have 37 Python runs and 45 forward passes left. Before doing anything else I will check what exists in the kernel and in `/work`, printing only short summaries. This run only gathers information, so it cannot confirm or refute a hypothesis. It should show whether the Stage-1 plan (reuse the unused z token to carry an amplified dose-residual signal) was already built and tested.

```python
import os, time, numpy as np
fs=sorted(os.listdir('/work'))
print([(f, time.strftime('%H:%M', time.localtime(os.path.getmtime('/work/'+f)))) for f in fs])
for f in ['setup.py','inj.py','tf.py','wm2.py','wm2_rollout.py']:
    src=open('/work/'+f).read().splitlines()
    defs=[l[4:].split('(')[0] for l in src if l.startswith('def ')]
    com=[l[:90] for l in src if l.strip().startswith('#')][:4]
    print('\n',f,len(src),'lines; defs:',defs[:20]); print('  comments:',com)
for f in ['s1data.npz','jac.npz','tf_train.npz']:
    d=np.load('/work/'+f, allow_pickle=True); print(f,{k:d[k].shape for k in d.files})
p='/work/edited'; print('edited path:', os.listdir(p)[:10] if os.path.isdir(p) else os.path.getsize(p))
print('edits:', edited(), 'z_head absmax', float(np.abs(load('z_head.weight')).max()))
g=[k for k,v in list(globals().items()) if not k.startswith('_') and not isinstance(v,type(os))]
print(len(g)); print(g[-120:])
```