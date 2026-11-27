I'm resuming after a restart. The last run (265) only measured activation scales and how much each hidden unit matters, presumably to find spare units to repurpose. My notebook ends at run 207, so runs 208 to 265 were not written down. Before using any of my remaining 35 Python runs on edits, I need to recover:
1. the earlier notebook sections: how I estimated the true effects from the records, and the NE (norepinephrine) truth;
2. what the newer /work files contain (chan.py, gen_wc.py, rollout_h2.py, evalh.py, wI_chan.npy, wI_head2.npy, corechan.npz).

This is only orientation, so it neither confirms nor refutes a hypothesis. My working belief is that the true insulin effect is a linear lag kernel in the dose change (x2 about -14 to -16.5 mg/dL), and the model's effect is about 6 times too weak.

```python
import os, time, re
txt = open('/work/notes.md').read()
print("notes len", len(txt))
unseen = txt[:max(0, len(txt)-8000)]
# headers with positions
for m in re.finditer(r'^#.*$', txt, re.M):
    print(m.start(), m.group(0)[:100])
print("=================== UNSEEN PART (capped 11000) ===================")
print(unseen[:11000])
print("=================== FILES ===================")
fl = []
for f in os.listdir('/work'):
    p = os.path.join('/work', f)
    st = os.stat(p)
    fl.append((st.st_mtime, f, st.st_size))
fl.sort()
t0 = fl[0][0]
for mt, f, sz in fl:
    print(f"{f:22s} {sz:>10d}  t+{(mt-t0)/60:7.1f} min")
for mt, f, sz in fl:
    if f.endswith('.py'):
        src = open('/work/'+f).read()
        defs = [l.strip()[:110] for l in src.splitlines() if l.startswith('def ') or (l.startswith('#') and len(l) > 3)]
        print(f"--- {f}: " + " | ".join(defs[:14]))
```