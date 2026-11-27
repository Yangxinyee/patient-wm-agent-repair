The write path is wired correctly in the state: writes are exactly 0 at non-drug bundles, and the cut and rewiring cost +0.0016 nats. The u-channel itself is wrong, though. u[4] misses 0.1×rate by up to 0.6, writes carry an intercept of −2.5, R² is only 0.55, and norepinephrine-only bundles get large writes. The cause: replacing one pre-norm output changed the encoder norm's mean and spread for every bundle, which distorted all of u.

The fix is an antisymmetric pair. Rewrite pre-norm outputs 4 and 2 as (their old average ± s), where s is a linear rate readout that includes the norm's mean term. This leaves the norm's mean exactly unchanged and its spread nearly so. The read uses u[4] − u[2] = 0.2·rate, and the bias pair cancels the decoder's constant. I'll also halve `cb` to keep the memory at about one unit per unit/h.

- **Confirms:** u[4]−u[2] tracks 0.2×rate with max error ≲0.05, the write R² is >0.98, and norepinephrine-only writes are ≈0.

```python
src2=r'''
def build_ins2(P0, dvec, d0, keep, jU=4, jR=2, jI=67, k3=20, kap=0.1, gc=1.0, cv=6., co=6., gE=5., cb=1., half_life_h=1.0):
    P=build_ins(P0, dvec, d0, keep, jU, jR, jI, k3, 0.1, cv, co, gE, cb, half_life_h)
    Q=G['cl'](P0)
    W2o=Q['encode_action.mlp2.2.weight']; b2o=Q['encode_action.mlp2.2.bias']; gao=Q['encode_action.norm.weight']; bao=Q['encode_action.norm.bias']
    kk=torch.as_tensor(np.asarray(keep)); dv=torch.as_tensor(np.asarray(dvec),dtype=torch.float32)
    coef=dv*gao[kk]; S=float(coef.sum())
    a=torch.full((192,),-S/192.); a[kk]+=coef
    s_w=kap*(a[:,None]*W2o).sum(0); s_b=kap*float((a*b2o).sum())
    m_w=(W2o[jU]+W2o[jR])/2; m_b=float(b2o[jU]+b2o[jR])/2
    W2=P['encode_action.mlp2.2.weight']; b2=P['encode_action.mlp2.2.bias']
    W2[jU]=m_w+s_w; b2[jU]=m_b+s_b; W2[jR]=m_w-s_w; b2[jR]=m_b-s_b
    c0=d0+float((dv*bao[kk]).sum())
    ga=P['encode_action.norm.weight']; ba=P['encode_action.norm.bias']
    ga[jU]=gc; ga[jR]=gc; ba[jU]=kap*gc*c0; ba[jR]=-kap*gc*c0
    return P
'''
open('/work/lib4.py','a').write(src2); exec(src2,G)
P2=build_ins2(PO,dvec,d0,keep)
ds2=reenc(dsE,P2)
du=(ds2['u'][:,jU]-ds2['u'][:,jR]).numpy()
print('u diff vs 0.2r: max err %.4f mean err %.4f | NEonly max %.4f'%(np.abs(du-0.2*rsI)[drug].max(), (du-0.2*rsI)[drug].mean(), np.abs(du[ne&~ins]).max()))
H2,O2,st2=tf_loop(P2,ds2,keep=('hf',))
wr=(H2[:,:,jI]-st2['hf'][:,:,jI]).numpy(); wm=wr.mean(1)
sl=np.polyfit(rsI[cl3],wm[cl3],1); res=wm[cl3]-np.polyval(sl,rsI[cl3])
print('write slope %.3f int %.3f R2 %.4f | nodrug max %.2e | NEonly p99 %.3f | endonly p99 %.3f'%(sl[0],sl[1],1-res.var()/wm[cl3].var(),np.abs(wr[~drug]).max(),np.percentile(np.abs(wm[ne&~ins]),99),np.percentile(np.abs(wm[ins&(rsI==0)]),99)))
n2_=tf_eval(P2,ds2)[0]; print('NLL diff %.5f'%(n2_.mean()-n0.mean()))
```