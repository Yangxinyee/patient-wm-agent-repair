The new mlp2 units are better chosen (importance ≈0.01 against a median of 0.33), and the empty-set u is unchanged (1e-6). But the encoder channel is still clearly nonlinear: insulin R² is 0.70 and 0.33, NE's slope is 2.15 against a designed 1.0, and it differs between 1- and 2-drug bundles.

I now see my mistake. I chose d̂ to be quiet only for the prelude tokens. The channel, though, enters x2 along d̂/(γ1·γ_enc) and enters u along d̂/γ1. Those are different directions, and they can align with high-variance directions of x2 and u. Each LayerNorm then rescales by a std that moves with the dose, which is state-dependent and nonlinear.

The fix is to pick d̂ that is jointly quiet: small for the prelude tokens and for its images in normalized x2 and normalized u, so neither LayerNorm's std responds to first order. The core numbers also explain the tiny analytic gain. `delta_in` softplus ≈ 0.99 and `b_in`/`c_out` norms are ≈4.5, so the path is real. The gain is small only because the slot stream has std ≈50 and the LayerNorms divide by it, so the channel just needs a larger absolute size there.

Success is encoder R² ≈ 1, slopes ≈ 0.3 and 1.0, and equal slopes for 1- and 2-drug bundles.

```python
X2a=np.vstack([E0['x2'],E0['xe'][None]]); xh=(X2a-X2a.mean(1,keepdims=True))/_lnstd(X2a)[:,None]
Mxh=xh.T@xh/len(xh)
Ut=Dd['X7'][:,5]; wh=(Ut-Ut.mean(1,keepdims=True))/_lnstd(Ut)[:,None]; Mwh=wh.T@wh/len(wh)
D2=np.diag(1/(g1_*ge_)); D3=np.diag(1/g1_)
Mtot=Mx_+D2@Mxh@D2+D3@Mwh@D3
Dh2,evt=_small_dirs(Mtot,Cc_,2)
for i in range(2):
    d=Dh2[:,i]; print('dir',i,'xn %.2e  x2hat %.2e  uhat %.2e'%(d@Mx_@d, d@D2@Mxh@D2@d, d@D3@Mwh@D3@d))
d_old=rep['Dh']; print('old dirs:',[('xn %.2e x2hat %.2e uhat %.2e'%(d_old[:,i]@Mx_@d_old[:,i], d_old[:,i]@D2@Mxh@D2@d_old[:,i], d_old[:,i]@D3@Mwh@D3@d_old[:,i])) for i in range(2)])
P['Dh']=Dh2
Wc,rep=design_build(W,Dd,P)
E1=enc_u({k:_np(v) for k,v in Wc.items()},XU,key); du=E1['u']-E0['u']
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    dd=Dh2[:,i]/g1_; dd=dd-dd.mean(); proj=du@dd/(dd@dd)
    for msk_,lab_ in [(E0['cnt']==1,'n1'),(E0['cnt']==2,'n2')]:
        lr_=np.polyfit(rr[msk_],proj[msk_],1); r2_=1-np.var(proj[msk_]-np.polyval(lr_,rr[msk_]))/np.var(proj[msk_])
        print(nm_,lab_,'slope %.4f int %.4f R2 %.5f'%(lr_[0],lr_[1],r2_))
xn_new=_ln(E1['u'],g1_,_np(W['core.prelude.0.norm1.bias'])); xn_old=_ln(E0['u'],g1_,_np(W['core.prelude.0.norm1.bias']))
dy=xn_new-xn_old
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    p_=dy@Dh2[:,i]; m_=E0['cnt']>=1; lr_=np.polyfit(rr[m_],p_[m_],1)
    print(nm_,'LN1-space channel slope %.4f R2 %.5f  off-channel |dy| %.4f'%(lr_[0],1-np.var(p_[m_]-np.polyval(lr_,rr[m_]))/np.var(p_[m_]),np.linalg.norm(dy-np.outer(dy@Dh2[:,0],Dh2[:,0])-np.outer(dy@Dh2[:,1],Dh2[:,1]),axis=1).mean()))
```