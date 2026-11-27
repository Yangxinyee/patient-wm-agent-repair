fwd_3 is a factual request and has no states file, so the loop stopped there. That was a bug in my code, not a problem with the data. fwd_2 is the sel1 tuning set.

The kernel still holds a lot from runs 96–104 that the notebook does not cover. This includes readout objects (RO_ins, RO_ne, RO_i, RO_n), saved original head weights (W0o, b0o, W2o, b2o, TRo) and helper functions (make_edit, head_run, fit_readout).

Next I will map each states file to its selection list. I will also print small summaries of those readout and edit objects, to see whether I had already built a candidate edit.

If those objects hold fitted readout weights and gains, I can go straight to an offline check of the edit. If not, I rebuild the readouts from the six training state batches.

```python
sel_tr=json.load(open('/work/sel_train.json')); sel_tr2=json.load(open('/work/sel_tr2.json')); ss=json.load(open('/work/states_stays.json'))
lists={'sel1':json.load(open('/work/sel1.json')),**{'trA_'+k:v for k,v in sel_tr.items()},**{f'tr2_{i}':v for i,v in enumerate(sel_tr2)},**{'ss_'+k:v for k,v in ss.items()}}
for k in [2,4,5,6,7,8,9]:
    z=np.load(f'/work/fwd_{k}/states_original.npz'); s=set(z['stay_id'].tolist())
    match=[n for n,l in lists.items() if set(l)==s]
    print(k,len(s),'matches',match)
def brief(o,depth=0):
    if hasattr(o,'shape'): return f"{type(o).__name__}{tuple(o.shape)}"
    if isinstance(o,dict): return '{'+', '.join(f"{k}:{brief(v,depth+1)}" for k,v in list(o.items())[:12])+'}'
    if isinstance(o,(list,tuple)): return f"{type(o).__name__}[{len(o)}]"+('('+', '.join(brief(x,depth+1) for x in o[:6])+')' if depth<2 else '')
    return repr(o)[:60]
for nm in ['RO_ins','RO_ne','RO_i','RO_n','ro','units','ED','ORIG','datasets','base_out','ACTS','ACT','CFG','cfg','consts','KEYS','CODE','gins','fs','rm','las','ns','ry','Gk','med','fsel','Ds','ALLR','ALLW','files','batches']:
    if nm in globals(): print(nm,'=',brief(globals()[nm])[:400])
```