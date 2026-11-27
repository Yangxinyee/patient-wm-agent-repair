I've just been restarted, and the notebook stops at the rollout-engine validation (around run 150). Runs 150 to 182 seem to have produced files the notes don't describe (oracle.py, cf.py, rollout_h.py, tf.py). Before I design an edit, I need to recover that state: the start of the notebook, the file list with modification times, and the variables still in the kernel. This run only reads; it changes nothing.

What I currently believe:
- The original model badly underestimates both drug effects. On the 100 tuning stays in sel1, insulin x2 gives -2.1 against my record-based estimate of -14.1, and norepinephrine x2 gives +1.1 against +4.7.
- My local rollout engine reproduces the tool closely.

The first step is to check whether the undocumented runs already went beyond this point.

```python
import os, time
txt=open('/work/notes.md').read()
print('notes length',len(txt))
print(txt[:max(0,len(txt)-8000)])
print('=====FILES=====')
for f in sorted(os.listdir('/work')):
    p='/work/'+f; st=os.stat(p)
    print(f, st.st_size, time.strftime('%d %H:%M:%S', time.localtime(st.st_mtime)), os.listdir(p) if os.path.isdir(p) else '')
print('=====GLOBALS=====')
print(sorted([k for k in globals().keys() if not k.startswith('_')]))
```