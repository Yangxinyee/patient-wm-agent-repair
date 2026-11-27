With the corrected units, my local z posterior now matches the model at insulin bundles: median error 0.005, worst dims (20, 24, 9) about 0.013. So the old binary-dims mismatch was the unit mapping, and my u, e and context vectors are right. The table at the end failed only on a formatting slip (the rate is stored as a (code, value) pair).

Next I will test whether a single linear readout of the prelude-normalised u token can recover the insulin start rate (and separately the norepinephrine rate) over the full range, including doubled and halved doses. Each readout must give 0 when that drug is absent or u is empty. The tuning grid covers insulin-only, norepinephrine-only, both-drug and empty u. I will also confirm that head 2 attends almost fully to the u token at every bundle type.

- **Confirms:** errors are small (sd well under 0.2 u/h for insulin) and head 2 attention stays near 1. A rank-1 edit of head 2 can then write the decoded rate into the slot tokens.
- **Refutes:** errors are large. Then the encoder must be changed instead.

```python
def u_of(ins=None, ne=None):
    act=[]
    if ins is not None: act.append(atom_vec('insulin',ins))
    if ne is not None: act.append(atom_vec('norepinephrine',ne))
    if not act: return torch.zeros(192,dtype=torch.float64)
    X_=torch.stack(act); n_=len(act)
    s_=mlp_t(X_,'encode_action.mlp1').sum(0)/math.sqrt(n_)+(torch.tensor([[math.log(1+n_)]],dtype=torch.float64)@Wt['encode_action.size.weight'].T+Wt['encode_action.size.bias'])[0]
    return LNt(mlp_t(s_[None],'encode_action.mlp2')[0],'encode_action.norm')
# sanity: compare with enc_eu on a real bundle
kk=ins_k[0]; act_=[a for a in B0[kk][1] if a['ek']==2 and a['kind']==0]; print('real bundle atoms', [(inv[a['code']],a['rate'][1]) for a in act_])
ri=[a['rate'][1] for a in act_ if a['code']==CODE['insulin']]; rn=[a['rate'][1] for a in act_ if a['code']==CODE['norepinephrine']]
print('u_of vs enc_eu max diff', float((u_of(ri[0] if ri else None, rn[0] if rn else None)-EU0[kk][1]).abs().max()))
gi=np.arange(0.25,12.01,0.25); gn=np.round(np.concatenate([np.linspace(0.005,0.1,8),np.linspace(0.12,2.0,40)]),4)
U=[];Ti=[];Tn=[]
for r in gi: U.append(u_of(ins=r)); Ti.append(r); Tn.append(0)
for r in gn: U.append(u_of(ne=r)); Ti.append(0); Tn.append(r)
for r in gi[::2]:
    for q_ in gn[::3]: U.append(u_of(ins=r,ne=q_)); Ti.append(r); Tn.append(q_)
U.append(u_of()); Ti.append(0); Tn.append(0)
U=torch.stack(U); L1=LNt(U,'core.prelude.0.norm1').numpy(); Ti=np.array(Ti); Tn=np.array(Tn)
Xd=L1; 
for nm_,T_ in [('insulin',Ti),('norepi',Tn)]:
    for lam in [1e-6,1e-4,1e-2]:
        A_=Xd.T@Xd+lam*np.eye(192); cvec=np.linalg.solve(A_,Xd.T@T_); pr=Xd@cvec; res=pr-T_
        print(nm_,'lam',lam,' resid sd %.4f  max %.3f  |c| %.1f'%(res.std(),np.abs(res).max(),np.linalg.norm(cvec)))
```