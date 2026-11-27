Run 248 confirmed the signatures of my local tools: rollout with hooks, exact-shift oracle, readout, truth rows and head logits. The list of globals was cut before the newest names. Python keeps globals in creation order, so the names after `bundle_rat` are the ones created in runs ~230–244. Those should include the decoder experiment behind the "base/wcm/wc" output. I'll print just those trailing names, plus the types and shapes of the arrays among them, so I can pick that experiment up instead of redoing it.

```python
gl=[k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,types.ModuleType)]
i0=[i for i,k in enumerate(gl) if k.startswith('bundle_rat')]
i0=i0[0] if i0 else len(gl)-120
tail=gl[i0:]
desc=[]
for k in tail:
    v=globals()[k]
    if isinstance(v,np.ndarray): desc.append(f"{k}{list(v.shape)}")
    elif torch.is_tensor(v): desc.append(f"{k}T{list(v.shape)}")
    elif isinstance(v,dict): desc.append(f"{k}{{{len(v)}}}")
    elif isinstance(v,(list,tuple)): desc.append(f"{k}[{len(v)}]")
    elif callable(v): desc.append(f"{k}()")
    else: desc.append(f"{k}:{type(v).__name__}")
print(len(tail)); print(' '.join(desc)[:4000])
```