**Where things stand.** The last output came from a check of my numpy copy of the model against the tool's outputs. Across 300 reading bundles the mean error was 0.056 and the largest was 0.139. That suggests the copy now matches closely, much better than the earlier MAE of 1.3, but I'm not certain which quantity it compared.

**A gap in the notebook.** The notebook ends after run 73, so runs 74–121 and forward passes 5–7 were never recorded. Before I design or calibrate an edit, I need to recover what those runs did and what files they left. My plan is to list the kernel's variables, list the files in the newer forward-pass folders, and look at the kernel's input history.

**Rules I will keep to.** Every edit will be computed from my own regressions on records or states. I will not run any gradient training or optimiser over the model's weights, and I will use only the permitted files and the tool.

**What would confirm the current plan.** The plan is a gated linear correction in the value head that reads a dose-residual direction, with its gains calibrated by dose_edit. It holds up if the kernel contains a working reconstruction (value head plus probe) that I can build on. If the history shows I abandoned that design, I'll rethink before spending forward passes.

```python
import os, json, types
for d in sorted(os.listdir('/work')):
    p='/work/'+d
    if os.path.isdir(p):
        print(d, [(f, os.path.getsize(p+'/'+f)) for f in os.listdir(p)])
    else:
        print(d, os.path.getsize(p))
for f in ['sel1.json','sel_states.json','new300.json']:
    x=json.load(open('/work/'+f)); print(f, type(x).__name__, (len(x) if hasattr(x,'__len__') else x), str(x)[:150])
groups={}
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if hasattr(v,'shape'): s=f"{k}{tuple(v.shape)}"; groups.setdefault('array',[]).append(s)
    elif callable(v) and not isinstance(v,type): groups.setdefault('func',[]).append(k)
    elif isinstance(v,(int,float,str,bool)): groups.setdefault('scalar',[]).append(f"{k}={str(v)[:25]}")
    elif isinstance(v,(list,tuple,dict,set)): groups.setdefault('coll',[]).append(f"{k}[{len(v)}]")
    else: groups.setdefault('other',[]).append(f"{k}:{type(v).__name__}")
for g,l in groups.items(): print('==',g, len(l)); print(', '.join(l)[:3000])
hist = globals().get('_ih') or globals().get('In')
print('history cells:', None if hist is None else len(hist))
if hist is not None:
    for i in range(max(0,len(hist)-4), len(hist)-1):
        print(f'--- cell {i} ---'); print(hist[i][:2500])
```