The local bundle lists line up exactly with the saved state rows: 54 bundles for 54 rows. The encoder returns 192-vectors for the bundle encoding e and the infusion encoding u.

Next I'll rebuild the prelude tokens and attention probabilities from the true saved previous states at every infusion bundle in the 12 stays. That gives the spread of each head's slot-to-u attention. A head whose weight on u is high and varies little (coefficient of variation under about 0.15) would be a usable carrier for an injected rate signal.

```python
def prev_state(i):
    x=X[i-1]; return torch.tensor(x[:192]), torch.tensor(x[192:960].reshape(4,192))
rows=[]; Pall=[]
for s in sids:
    s=int(s); B=stay_bundles(s); idx=np.where(D['stay_id']==s)[0]
    for k in range(1,len(B)):
        i=idx[k]
        if not (D['has_insulin'][i] or D['has_norepinephrine'][i]): continue
        o_,h_=prev_state(i); dt=float(B[k].minute.iloc[0]-B[k-1].minute.iloc[0])*60
        e,u,_,_=enc_bundle(B[k]); z=zpost_mean(o_,h_,dt,e,u)
        P=pre_probs(pre_tokens(h_,dt,e,u,z)); P=np.asarray(P)
        Pall.append(P); rows.append(i)
print(len(rows), Pall[0].shape)
P=np.stack(Pall)
if P.ndim==4 and P.shape[-1]==7:
    a=P[:,:,:4,5]
    for hh in range(a.shape[1]): print(hh,'mean %.3f sd %.3f min %.3f'%(a[:,hh].mean(),a[:,hh].std(),a[:,hh].min()))
```