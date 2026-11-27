The last run printed the architecture but was cut off before the fwd_8 check finished. Facts worth keeping:
- The state update uses the posterior mean of z, so my writer's shift of z18/z28 on drug bundles is deterministic in the rollout too.
- The heads use z drawn from the prior.
- The value head is a 17-class softmax over fine bins. The predicted mean is computed from the bin centres.

Next I'll redo the fwd_8 check, keeping the output short, and list only the kernel objects I made later (functions and their signatures). That tells me what tools I still have.

What I expect: on insulin bundles, dz18 ≈ 0.29×rate. On norepinephrine bundles, dz28 ≈ 1.83×rate. Other dims should move only by small propagated amounts. If dz18 doesn't track the insulin rate, the writer reconstruction is wrong.

```python
import inspect
with open('/work/notes.md','a') as fh:
    fh.write("\n## Restart 3 (after run 149)\n- fwd8 = states BOTH (writer edit z_post 55->z18 ins, 109->z28 ne) on sel_states 99 train stays.\n"
             "- ARCH recap: state update uses z POSTERIOR MEAN; heads use z drawn from PRIOR (4 draws, probs averaged). value head = 17-class softmax (0 none, 1..16 fine bins), mean=sum p*centre.\n")
names=list(globals().keys()); i0=names.index('ss') if 'ss' in names else 0
lst=[]
for k in names[i0:]:
    v=globals()[k]
    if k.startswith('_') or isinstance(v,types.ModuleType): continue
    if callable(v) and not hasattr(v,'shape'):
        try: lst.append(k+str(inspect.signature(v)))
        except Exception: lst.append(k+'()')
    elif hasattr(v,'shape'): lst.append(k+str(tuple(v.shape)))
    elif isinstance(v,(list,dict,tuple)): lst.append(k+':'+type(v).__name__+str(len(v)))
print(' | '.join(lst)[:4500])
print('X==X8o maxdiff', np.abs(X8o-X).max())
sid8=d8o['stay_id']; hr8=d8o['hour']; hi=d8o['has_insulin'].astype(bool); hn=d8o['has_norepinephrine'].astype(bool)
print('hours sample', hr8[:8])
ri=np.array([ins_all[sidx[int(s)], int(np.floor(h))] if 0<=np.floor(h)<25 else np.nan for s,h in zip(sid8,hr8)])
rn=np.array([ne_all[sidx[int(s)], int(np.floor(h))] if 0<=np.floor(h)<25 else np.nan for s,h in zip(sid8,hr8)])
dz=X8e[:,960:992]-X8o[:,960:992]
for nm,m,dim,rr in [('ins',hi,18,ri),('ne',hn,28,rn)]:
    sl=np.polyfit(rr[m],dz[m,dim],1); cc=np.corrcoef(rr[m],dz[m,dim])[0,1]
    oth=np.delete(np.abs(dz[m]).mean(0),[18,28])
    print(nm,'n',m.sum(),'slope,icpt',np.round(sl,3),'corr',round(cc,3),'mean|dz| other dims max',round(oth.max(),4),'mean',round(oth.mean(),4))
m0=~hi&~hn; print('nondrug mean|dz| z18,z28,max other', np.round(np.abs(dz[m0]).mean(0)[[18,28]],4), round(np.delete(np.abs(dz[m0]).mean(0),[18,28]).max(),4))
```