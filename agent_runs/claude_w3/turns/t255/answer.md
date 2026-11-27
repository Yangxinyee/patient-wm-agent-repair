At infusion bundles, prelude head 2 puts almost all of the slots' attention on token 5 (mean 0.990, sd 0.035, min 0.65). Heads 0, 1 and 3 put exactly zero on it, which looks like masking or a different token order.

If token 5 is the u token, head 2 is a nearly fixed carrier. Its value vector for u could deliver a rate signal into the slots with a weight close to 1. I'll check the full slot attention profile per head, both at infusion bundles and at bundles without infusions, to confirm which token is u and what head 2 does when no infusion is present.

```python
print('action bundles: mean slot->token attention per head')
print(np.round(P[:,:,:4,:].mean((0,2)),3))
Q=[]
for s in sids[:4]:
    s=int(s); B=stay_bundles(s); idx=np.where(D['stay_id']==s)[0]
    for k in range(1,len(B)):
        i=idx[k]
        if D['has_insulin'][i] or D['has_norepinephrine'][i]: continue
        o_,h_=prev_state(i); dt=float(B[k].minute.iloc[0]-B[k-1].minute.iloc[0])*60
        e,u,_,_=enc_bundle(B[k]); z=zpost_mean(o_,h_,dt,e,u)
        Q.append(np.asarray(pre_probs(pre_tokens(h_,dt,e,u,z))))
Q=np.stack(Q); print('non-action', len(Q)); print(np.round(Q[:,:,:4,:].mean((0,2)),3))
```