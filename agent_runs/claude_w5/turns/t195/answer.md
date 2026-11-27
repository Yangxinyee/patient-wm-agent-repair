The dose can be read cleanly from one place. A linear readout of the action-encoder output u recovers the insulin and norepinephrine rates, including the x2 and x1/2 variants, with held-out R² 0.98 and 0.97. By the time the dose reaches the slots, it reads at only about 0.70 and 0.56.

**Proposed split of the gate:**
- **Dose part:** a rank-one addition to the prelude attention's value path that reads the rate from the u token. It writes a tiny "rate" signal into the slots, and is exactly 0 for the empty-drug token.
- **Centring part (expected dose given the state):** about three of the smallest prelude-FFN units, GELU-gated by the drug/no-drug score, which separates perfectly. They would compute "rate signal − linear estimate of expected dose" only at drug bundles.

The critical check is whether the expected dose can be read linearly from the prelude input n3. To keep the readout from simply copying the current rate, I fit it with the dose variants given the same target.

High held-out R² for both drugs would make this design workable.

```python
Xc=[];Yc=[];Fc=[]
for d in D:
    if not isdrug(d): continue
    for sfx,_,_ in VK:
        Xc.append(d['n3'+sfx].double()); Yc+= [[d['cI'],d['cN']]]*4; Fc+=[d['sid'] in FIT]*4
Xc=torch.cat(Xc); Yc=torch.tensor(Yc,dtype=torch.float64); Fc=torch.tensor(Fc)
print('n3 per-slot -> c_hat R2',ridge_val(Xc,Yc,Fc,[1e-3,1e-2,1e-1]))
Xcm=Xc.reshape(-1,4,192).mean(1); Ycm=Yc[::4]; Fcm=Fc[::4]
print('n3 slot-mean -> c_hat R2',ridge_val(Xcm,Ycm,Fcm,[1e-3,1e-2,1e-1]))
print('sd cI %.3f cN %.3f'%(float(Ycm[:,0].std()),float(Ycm[:,1].std())))
```