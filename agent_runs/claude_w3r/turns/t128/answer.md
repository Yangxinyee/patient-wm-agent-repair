Only u carries the dose cleanly: it predicts the start rate with R² 0.96. E (0.63) and `out` (0.59) carry rate information only through the glucose band, and the deviation from policy not at all (R² near 0 or below). So the strong u→E direction (gain 1.48) is not the direction in which u varies with rate.

The channel therefore has to be built from scratch:
- Make encode_action write the insulin rate along a chosen u-direction.
- Pick that direction so its response in `out` stands out from normal variation. Earlier I found a best-case signal-to-noise of about 3.6 noise SDs per unit of u perturbation, which may be enough without amplifying anywhere else.
- Read it at the glucose head through spare (dead) hidden units.

Building this needs unused hidden units in encode_action's two MLPs. Next I will find units that never activate. I feed mlp1 every insulin and norepinephrine start atom over the rate grid, including doubled and halved rates. I feed mlp2 the pooled vectors of all action bundles in the 100 selS stays, recorded and edited.

- **Confirms:** several units stay far negative (pre-activation below −3) on every input, so they can be repurposed safely.
- **Refutes:** no such units exist. Then I would have to edit live units and accept some side effects.

```python
W_m1=Wt['encode_action.mlp1.0.weight']; b_m1=Wt['encode_action.mlp1.0.bias']; W_m2=Wt['encode_action.mlp2.0.weight']; b_m2=Wt['encode_action.mlp2.0.bias']
def atom_vec(code,rate,kind=0):
    a=dict(code=CODE[code],kind=kind,ek=2,unit=OPTS['unit'][code],ru=OPTS['ru'][code],val=None,dose=(code,rate),rate=(code,rate))
    return mm.embed_atom(a,OPTS).double()
# check dose==rate in data
print('dose/rate ratio insulin', (recs[recs.event=='insulin'].dose/recs[recs.event=='insulin'].rate).describe()[['mean','std']].round(3).to_dict(),
      ' norepi', (recs[recs.event=='norepinephrine'].dose/recs[recs.event=='norepinephrine'].rate).describe()[['mean','std']].round(3).to_dict())
gr_ins=np.unique(np.concatenate([np.arange(0.25,12.01,0.25)])); gr_ne=np.unique(np.concatenate([np.linspace(0.005,2.0,200)]))
A_atoms=torch.stack([atom_vec('insulin',r) for r in gr_ins]+[atom_vec('norepinephrine',r) for r in gr_ne])
pre1=A_atoms@W_m1.T+b_m1; mx1=pre1.max(0).values.numpy()
print('mlp1: units with max pre-act < -3:', int((mx1<-3).sum()), ' < -5:', int((mx1<-5).sum()), ' (of 768)')
# mlp2 inputs: pooled s for action bundles (recorded and edited x2, x0.5)
S_list=[]
for s in selS:
    ev_=recs[recs.stay_id==s].sort_values('hour')
    for f_ins,f_ne in [(1,1),(2,1),(0.5,1),(1,2),(1,0.5)]:
        ev2=ev_.copy(); mI=(ev2.event=='insulin')&(ev2.hour>=12); mN=(ev2.event=='norepinephrine')&(ev2.hour>=12)
        ev2.loc[mI,['rate','dose']]*=f_ins; ev2.loc[mN,['rate','dose']]*=f_ne
        for t_,al in build_bundles(ev2,OPTS):
            act=[a for a in al if a['ek']==2 and a['kind']==0]
            if act:
                X_=torch.stack([mm.embed_atom(a,OPTS).double() for a in act]); n_=len(act)
                s_=mlp_t(X_,'encode_action.mlp1').sum(0)/math.sqrt(n_)+ (torch.tensor([[math.log(1+n_)]],dtype=torch.float64)@Wt['encode_action.size.weight'].T+Wt['encode_action.size.bias'])[0]
                S_list.append(s_)
S2=torch.stack(S_list); pre2=S2@W_m2.T+b_m2; mx2=pre2.max(0).values.numpy()
print('n pooled vectors', S2.shape[0], ' mlp2: units max pre-act < -3:', int((mx2<-3).sum()), ' < -5:', int((mx2<-5).sum()))
dead1=np.where(mx1<-3)[0]; dead2=np.where(mx2<-3)[0]; print('dead1 sample', dead1[:10], ' dead2 sample', dead2[:10])
```