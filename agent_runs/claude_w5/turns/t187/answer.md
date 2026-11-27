The fitting data built in 31 s:
- 5,294 bundles, 2,393 of them drug bundles;
- 1,840 drug bundles have a preceding glucose finger reading, giving glucose directions;
- 2,291 have a preceding MAP reading, giving MAP directions.

Next I will check the two things the design relies on:
1. **Consistency of the belief direction.** I compare each bundle's h_new change with the mean direction, overall and per slot. If they are similar, one fixed write direction is enough.
2. **Head response.** Using the next reading's row, I compute how much the predicted glucose (MAP) mean moves under the state change alone, per unit reading change. First I confirm how the predicted mean is formed from `probs_from_v0` by matching the stored targets.

High cosines and a stable, sizeable head response would confirm that the belief-injection design is workable.

```python
idx={}
for i,hh in enumerate(H): idx[(hh['sid'],hh['k'],hh['code'])]=i
ii=torch.arange(0,400)
Pp=probs_from_v0(V0[ii],fixed[ii],TRr[ii],nz=4,seed=0)
Pp=Pp if torch.is_tensor(Pp) else torch.tensor(np.asarray(Pp))
tg=torch.tensor([H[i]['target'] for i in ii.tolist()])
m1=(Pp[:,1:]*cen[ii]).sum(1); m2=m1/Pp[:,1:].sum(1)
print('P shape',tuple(Pp.shape),'rmse m1 %.2f m2 %.2f'%(float(((m1-tg)**2).mean().sqrt()),float(((m2-tg)**2).mean().sqrt())))
for nm in ['vg','vm']:
    V=torch.stack([d[nm] for d in D if nm in d]).double(); mv=V.mean(0)
    cosall=Fn.cosine_similarity(V.reshape(len(V),-1),mv.reshape(1,-1)).numpy()
    cslot=[float(Fn.cosine_similarity(mv[i],mv[j],dim=0)) for i,j in [(0,1),(0,2),(0,3),(1,2)]]
    print(nm,'n',len(V),'|mean| %.3f'%float(mv.norm()),'cos to mean q10/50/90',np.round(np.quantile(cosall,[.1,.5,.9]),2),'slot-pair cos',np.round(cslot,2))
```