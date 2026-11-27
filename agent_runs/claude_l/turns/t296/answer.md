I'm resuming after a restart. The notebook has no entries after run 177, but the kernel kept its variables. The last result (run 272) was a check of two hidden channels ("ch8", "ch9") that are meant to detect insulin and norepinephrine and decode their dose. Detection is clean: the present/absent ranges don't overlap. The dose decode is imprecise: p99 error 0.25 for insulin and 0.13 for norepinephrine, with occasional errors near 1.

I have only 28 Python runs left, so I can't rebuild from memory. First I'll inventory the kernel's variables and dump its input history to /work so I can see what runs 177–272 built. Then I'll record that in the notebook. If the history exists, I can resume the planned z-token edit directly. If not, I'll have to reconstruct it from the variables and the /work scripts.

```python
import os, types
print(sorted(os.listdir('/work')))
for f in ['inj.py','wm2.py','wm2_rollout.py','setup.py','tf.py','notes.md']:
    p='/work/'+f
    if os.path.exists(p): print(f, os.path.getsize(p), end=' | ')
print()
print('edited:', edited())
H = globals().get('In') or globals().get('_ih')
print('history:', type(H).__name__, (len(H) if H is not None else None))
if H:
    with open('/work/history_dump.py','w') as fh:
        for i,c in enumerate(H): fh.write(f"\n# ==== cell {i} ====\n{c}\n")
    n=len(H)
    for i in range(max(0,n-50), n-2):
        c=H[i].strip().replace('\n',' ; ')
        print(i, c[:110])
    for i in range(max(0,n-2), n):
        print(f'===== cell {i} =====')
        print(H[i][:2500])
g = {k: v for k, v in list(globals().items()) if not k.startswith('_') and k not in ('In','Out','H','g')}
data_items, fns = [], []
for k, v in g.items():
    if isinstance(v, types.ModuleType): continue
    if isinstance(v, np.ndarray): data_items.append(f"{k}:nd{v.shape}")
    elif torch.is_tensor(v): data_items.append(f"{k}:t{tuple(v.shape)}")
    elif isinstance(v, dict): data_items.append(f"{k}:dict{len(v)}")
    elif isinstance(v, (list, tuple)): data_items.append(f"{k}:{type(v).__name__}{len(v)}")
    elif isinstance(v, (int, float, str, bool)): data_items.append(f"{k}={str(v)[:30]}")
    elif callable(v): fns.append(k)
    else: data_items.append(f"{k}:{type(v).__name__}")
print('DATA:', ' | '.join(data_items)[:5000])
print('FNS:', ', '.join(fns)[:2500])
```