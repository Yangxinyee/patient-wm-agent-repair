The rebuild of q is exact: max|q − Q99| = 0.0000. Here `k` is the state row of the last bundle before the reading, and z is set to the prior mean. So I can now compute q, and Δq, from any state file, including the edited model's.

Next is the writer side. I need u_k for drug bundles from my own action set encoder, applied to the atom vectors from `drug_atoms`:

u = LN(mlp2(Σ mlp1(atom)/√n + size(log(1+n)))).

I'll validate it the only way available. For drug bundles in the 99-stay state file, rebuild the posterior mean z_post([ctx, e_empty, u]) and compare it with the stored posterior means, trying the END-atom handling modes that `drug_atoms` accepts.

An error comparable to the earlier 0.05 would confirm the u_k rebuild, and I can then fit the dose readout p·u. If the error is large, END atoms are probably routed to the measurement encoder, and I'll test that split.

```python
def setenc_my(V,pre):
    V=np.atleast_2d(np.asarray(V,float)); n=len(V)
    s=(mlpf(V,pre+'.mlp1').sum(0)/np.sqrt(n) if n>0 else 0)+(np.log1p(n)*L(pre+'.size.weight')[:,0]+L(pre+'.size.bias'))
    return LNf(mlpf(s,pre+'.mlp2'),L(pre+'.norm.weight'),L(pre+'.norm.bias'))
e0_my=setenc_my(np.zeros((0,192)),'encode'); print('e_empty match', np.abs(e0_my-np.asarray(e_empty(),float)).max())
sid2=S2['stay_id']; hr2=S2['hour']; drg=(S2['has_insulin']|S2['has_norepinephrine']).astype(bool)
rows_d=[j for j in np.where(drg)[0] if j>0 and sid2[j-1]==sid2[j]][:300]
for mode in ['both','u','e','none']:
    try:
        er=[]
        for j in rows_d:
            dt=(hr2[j]-hr2[j-1])*3600; _,ctx,_=view_from(X2[j-1],dt)
            V=drug_atoms(sid2i[int(sid2[j])], int(np.floor(hr2[j])), mode)
            u=setenc_my(V,'encode_action') if len(V)>0 else np.zeros(192)
            zz=mlpf(np.concatenate([ctx[0],e0_my,u]),'z_post')[:32]; er.append(np.abs(zz-X2[j,960:992]).mean())
        print(mode,'n',len(er),'mean |dz| %.4f max %.4f'%(np.mean(er),np.max(er)))
    except Exception as ex: print(mode,'fail',repr(ex)[:120])
```