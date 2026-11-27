The exact null-space idea fails: the prelude tokens have only a 1-dimensional null space, and my two centring constraints remove it. The earlier "zero" eigenvalues were just very small, so quiet but not null directions are what we have. That is acceptable, since anything leaking along them is tiny.

The LayerNorm gains are moderate (γ1 0.86–1.18, γ_enc 0.56–0.99), so dividing by them was not what made the encoder channel nonlinear. The real flaw is in my mlp2 gating. The on-unit's pre-activation includes G·(gate − 1), and the gate is only ≈±1 after the edit, because the repurposed mlp1 units slightly change the pooled vector. Any deviation leaks into the channel with gain G = 12. That fits a correct slope (0.30) with R² only 0.5.

Fix: pair each gated channel unit with a gate-only twin that has the same gate and no channel, and subtract their outputs. The gate term then cancels exactly whenever both are on. That needs 8 mlp2 units, which I'll now choose by actual importance (activation sd × output-column norm) rather than peak activation. I also print the core gains (`delta_in` softplus, `b_in`, `c_out` norms) to understand the tiny analytic gain into `out`.

Success is encoder R² ≈ 1, with slopes ≈ 0.3 (insulin) and 1.0 (NE) the same for 1- and 2-drug bundles, and the empty-set u unchanged.

```python
code_chan=code_chan.replace("""    for (uA,uB),c,m,dn in [((P['u2'][0],P['u2'][1]),c1I,mI,'I'),((P['u2'][2],P['u2'][3]),c1N,mN,'N')]:
        W3[uA]=beta*c+G*g; b3[uA]=B-beta*m+G*g0-G
        W3[uB]=np.sqrt(2)*beta*c-G*g; b3[uB]=B-np.sqrt(2)*beta*m-G*g0-G
        W4[:,uA]=c2[dn]; W4[:,uB]=c2[dn]; b4-=c2[dn]*B""",
"""    for us,c,m,dn in [(P['u2'][0:4],c1I,mI,'I'),(P['u2'][4:8],c1N,mN,'N')]:
        uA,uA2,uB,uB2=us
        W3[uA]=beta*c+G*g; b3[uA]=B-beta*m+G*g0-G; W4[:,uA]=c2[dn]
        W3[uA2]=G*g; b3[uA2]=B+G*g0-G; W4[:,uA2]=-c2[dn]
        W3[uB]=np.sqrt(2)*beta*c-G*g; b3[uB]=B-np.sqrt(2)*beta*m-G*g0-G; W4[:,uB]=c2[dn]
        W3[uB2]=-G*g; b3[uB2]=B-G*g0-G; W4[:,uB2]=-c2[dn]""")
open('/work/chan.py','w').write(code_chan); exec(code_chan)
O_={k:_np(v) for k,v in W.items()}
a1=_gelu(XU@O_['encode_action.mlp1.0.weight'].T+O_['encode_action.mlp1.0.bias'])
imp1=a1.std(0)*np.linalg.norm(O_['encode_action.mlp1.2.weight'],axis=0)
a2=_gelu(E0['pool']@O_['encode_action.mlp2.0.weight'].T+O_['encode_action.mlp2.0.bias'])
a2e=_gelu(E0['pe']@O_['encode_action.mlp2.0.weight'].T+O_['encode_action.mlp2.0.bias'])
imp2=np.sqrt(((a2-a2e)**2).mean(0))*np.linalg.norm(O_['encode_action.mlp2.2.weight'],axis=0)
u1s=np.argsort(imp1)[:4].tolist(); u2s=np.argsort(imp2)[:8].tolist()
print('mlp1 units',u1s,np.round(imp1[u1s],4),'median imp',round(float(np.median(imp1)),3)); print('mlp2 units',u2s,np.round(imp2[u2s],4),'median',round(float(np.median(imp2)),3))
P=dict(u1=u1s,u2=u2s,s1=1.0,beta=1.0,B=4.0,G=12.0,kapI=0.3,kapN=1.0,gv=1.0,tgtI=0.3,tgtN=1.0,g3=[1.0,1.0])
Wc,rep=design_build(W,Dd,P)
print('gate_err',rep['gate_err'],'ev_xn',rep['ev_xn'],'ev_v',rep['ev_v'],'sJ',rep['sJ'])
E1=enc_u({k:_np(v) for k,v in Wc.items()},XU,key); du=E1['u']-E0['u']
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    dd=rep['Dh'][:,i]/g1_; dd=dd-dd.mean(); proj=du@dd/(dd@dd)
    for msk_,lab_ in [(E0['cnt']==1,'n1'),(E0['cnt']==2,'n2')]:
        lr_=np.polyfit(rr[msk_],proj[msk_],1); r2_=1-np.var(proj[msk_]-np.polyval(lr_,rr[msk_]))/np.var(proj[msk_])
        print(nm_,lab_,'slope %.4f int %.4f R2 %.5f'%(lr_[0],lr_[1],r2_))
print('mean |du| %.4f |u| %.2f empty-u change %.6f'%(np.linalg.norm(du,axis=1).mean(),np.linalg.norm(E0['u'],axis=1).mean(),np.abs(E1['ue']-E0['ue']).max()))
spv_=np.log1p(np.exp(_np(W['core.delta_in']))); print('delta_in shape',SHAPES['core.delta_in'],'sp mean %.4f min %.4f max %.4f'%(spv_.mean(),spv_.min(),spv_.max()),
      '|b_in| %.2f |c_out| %.2f'%(np.linalg.norm(_np(W['core.b_in.weight']),2),np.linalg.norm(_np(W['core.c_out.weight']),2)))
```