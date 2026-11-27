fwd_4 to fwd_9 are all "states" requests on 100 training stays each, about 600 training stays in total. fwd_4 (4629 bundles) is the batch I had loaded as XA, and fwd_5 is batch0 from my notes. The printout was cut before the rest of the kernel variables and the lib.py function list.

Next I will show the remainder of that list compactly. I will also map each fwd_k states file to the selection files (sel_train.json, sel_tr2.json, states_stays.json). That tells me which training stays already have model states and which are still unused for a held-out check.

This settles whether I already hold enough state data, about 600 training stays, to fit a dose-deviation readout without spending more forward passes.

```python
import types
va=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,types.ModuleType) or (callable(v) and not hasattr(v,'shape') and not isinstance(v,(dict,list))): continue
    if hasattr(v,'shape'): va.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v,(dict,list,tuple)): va.append(f"{k}[{len(v)}]")
names=[x.split('(')[0].split('[')[0] for x in va]
i0=names.index('both_fl') if 'both_fl' in names else 0
print('VARS from both_fl:', ' '.join(va[i0:]))
src=open('/work/lib.py').read().split('\n')
print('lib.py defs:', [l[4:l.find('(')] for l in src if l.startswith('def ')])
for fn in ['sel_train.json','sel_tr2.json','states_stays.json','sel1.json']:
    o=json.load(open('/work/'+fn))
    if isinstance(o,dict): print(fn,'dict',{k:(len(v) if hasattr(v,'__len__') else v) for k,v in o.items()})
    else: print(fn,'list',len(o), o[:3] if not isinstance(o[0],list) else [len(x) for x in o])
sets={}
for k in range(2,10):
    z=np.load(f'/work/fwd_{k}/states_original.npz'); sets[k]=sorted(set(z['stay_id'].tolist()))
    print(k, len(sets[k]), sets[k][:3])
```