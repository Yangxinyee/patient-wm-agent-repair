`fu` has mean NLL 1.6382, so it is the u=0 run from pass 15, not the original model. That comparison is invalid. I'll find a pass directory that holds the original model's factual output on the same 100 stays and redo the split.

```python
for k in range(1,24):
    d='/work/fwd_%d'%k; fs_=os.listdir(d)
    if any('factual' in f_ for f_ in fs_): print(k, fs_)
```