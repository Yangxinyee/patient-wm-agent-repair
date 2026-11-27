The construction mostly works. Pooled space, prelude-token space and head-2 value space all have exactly zero-variance directions (eigenvalues 0), so the channel can be made invisible to everything except its intended reader. The set-size gate is exact (error 1e-7).

The encoder half is not yet linear, though. The insulin slope is right (0.30) but R² is only 0.50, and the NE slope is 1.53 (designed 1.0) and drops to 0.87 in two-drug bundles. The likely cause is that I divided the target direction by the two LayerNorm gains. Where those gains are small this blows up the perturbation of x2, so the LayerNorm's std changes with dose and the channel goes nonlinear.

The fix is to pick d̂ inside the exact null space of the prelude tokens, but minimising its size after division by the gains, so the perturbation stays small relative to |x2| (≈350) and |u| (≈12). Separately, the analytic gain from slots to `out` came out implausibly tiny (0.01, 0.00). So that path probably isn't dominated by `b_in`, and I'll print its components rather than trust it.

Success here is encoder R² ≈ 1 with slopes ≈ the designed κ (0.3 insulin, 1.0 NE) and equal for 1- and 2-drug bundles. I fixed the last print's bug.

```python
code_chan=code_chan.replace("    Cc=np.c_[1/g1,1/(g1*ge)]; Dh,evx=_small_dirs(Mx,Cc,2); rep['ev_xn']=evx",
"""    Cc=np.c_[1/g1,1/(g1*ge)]
    if 'Dh' in P: Dh=P['Dh']; rep['ev_xn']=np.diag(Dh.T@Mx@Dh)
    else: Dh,evx=_small_dirs(Mx,Cc,2); rep['ev_xn']=evx""")
code_chan=code_chan.replace("        g3=tgt/(Sj[i]*P['gv']*kap); blk+=g3*np.outer(c3[:,i],vh[:,i])",
"""        if 'c3' in P: c3=P['c3']
        g3=P['g3'][i] if 'g3' in P else tgt/(Sj[i]*P['gv']*kap); blk+=g3*np.outer(c3[:,i],vh[:,i])""")
open('/work/chan.py','w').write(code_chan); exec(code_chan)
g1_=_np(W['core.prelude.0.norm1.weight']); ge_=_np(W['encode_action.norm.weight'])
Mx_=np.einsum('ntd,nte->de',Dd['xn'],Dd['xn'])/(Dd['xn'].shape[0]*7)
evm,evecm=np.linalg.eigh(Mx_); N0=evecm[:,evm<1e-9*evm.max()]; print('null dim of prelude tokens', N0.shape[1], 'gamma1 min/max %.3f %.3f ge min/max %.3f %.3f'%(g1_.min(),g1_.max(),ge_.min(),ge_.max()))
Cc_=np.c_[1/g1_,1/(g1_*ge_)]
Bn=N0@np.linalg.svd(Cc_.T@N0)[2][2:].T          # null(Mx) ∩ constraints
Q=np.diag(1/(g1_*ge_)**2+1/g1_**2)
eq,evq=np.linalg.eigh(Bn.T@Q@Bn); Dh_=Bn@evq[:,:2]
print('Q-eig', np.round(eq[:3],3), '|dh/(g1ge)|', np.round(np.linalg.norm(Dh_/(g1_*ge_)[:,None],axis=0),3), '|dh/g1|', np.round(np.linalg.norm(Dh_/g1_[:,None],axis=0),3), 'constraint', np.round(Cc_.T@Dh_,6).ravel())
P.update(Dh=Dh_, g3=[1.0,1.0])
Wc,rep=design_build(W,Dd,P)
print('ev_xn', rep['ev_xn'], 'ev_v', rep['ev_v'], 'sJ', rep['sJ'])
E1=enc_u({k:_np(v) for k,v in Wc.items()},XU,key); du=E1['u']-E0['u']
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    dd=Dh_[:,i]/g1_; dd=dd-dd.mean(); proj=du@dd/(dd@dd)
    for msk_,lab_ in [(E0['cnt']==1,'n1'),(E0['cnt']==2,'n2')]:
        lr_=np.polyfit(rr[msk_],proj[msk_],1); r2_=1-np.var(proj[msk_]-np.polyval(lr_,rr[msk_]))/np.var(proj[msk_])
        print(nm_,lab_,'slope %.4f int %.4f R2 %.5f'%(lr_[0],lr_[1],r2_))
oth=du-np.outer(du@(Dh_[:,0]/g1_-np.mean(Dh_[:,0]/g1_)),np.zeros(192))
print('mean |du| %.4f  |u| %.2f  empty-u change %.6f'%(np.linalg.norm(du,axis=1).mean(),np.linalg.norm(E0['u'],axis=1).mean(),np.abs(E1['ue']-E0['ue']).max()))
spv_=np.log1p(np.exp(_np(W['core.delta_in']))); print('delta_in shape',SHAPES['core.delta_in'],'sp mean %.4f min %.4f max %.4f'%(spv_.mean(),spv_.min(),spv_.max()),
      '|b_in| %.2f |c_out| %.2f'%(np.linalg.norm(_np(W['core.b_in.weight']),2),np.linalg.norm(_np(W['core.c_out.weight']),2)))
```