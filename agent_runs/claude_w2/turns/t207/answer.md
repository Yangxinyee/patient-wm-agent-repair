The channel looks feasible. Heads 0 and 2 attend 100% to u at every bundle that has a drug (insulin, norepinephrine or both). At bundles without drugs, head 2 still attends 95% to the empty u. Head 0 instead splits between the slots (51%) and e (40%).

Decoding the insulin rate:

| Source | R² | Error sd (U/h) | Arm slopes |
|---|---|---|---|
| LN1(u) | 0.9986 | 0.06 | 0.97–1.0 |
| Head 0 + head 2 values | 0.991 | 0.15 | 0.95–1.0 |
| Head 2 alone | 0.94 | 0.31 | |

Both rows decode about 0 for empty u.

So a rank-1 addition to the prelude attention's output projection can write c·(insulin rate)·v into every slot with an error of about 0.15 U/h. That is far cleaner than anything decodable at the head now. The decoder w reads head 0 and head 2 outputs, and v is a chosen direction.

This run builds a first version and measures two things:
- **Precision:** fit w on head 0+2 values (target = rate at drug bundles, 0 at no-drug bundles), using each slot's actual attention-weighted head-0 output there, since head 0 does not attend to u in those bundles. Report the channel error for each bundle type.
- **Transmission and side effects:** put the edit into a copy of the weights, with v a random unit direction (mean-free) and amplitude c set from the measured slot-residual norm. Run teacher-forced on the 300 stays and compare the head's with_code at glucose rows with the unedited model. Check how much of the change the lag-1 and lag-2 doses explain, and the factual value NLL change, which must stay near zero because nothing reads the channel yet.

```python
capN=[]
def hpre3(k,act_n,hf,ctx,e,u,z,c):
    ar=c['seq_arm'][act_n]; s=c['seq_si'][act_n]
    m=(ar==0)
    if not m.any(): return
    mi=torch.as_tensor(np.where(m)[0])
    X=torch.cat([hf[mi]+W['core.slots'][None],e[mi][:,None],u[mi][:,None],F.linear(z[mi],W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    xn=LN(X,W,'core.prelude.0.norm1'); q=F.linear(xn,W['core.prelude.0.attn.q.weight']); kv=F.linear(xn,W['core.prelude.0.attn.kv.weight'])
    B=X.shape[0]; qh=q.reshape(B,7,4,48).transpose(1,2); kh=kv[...,:192].reshape(B,7,4,48).transpose(1,2); vh=kv[...,192:].reshape(B,7,4,48).transpose(1,2)
    o=(((qh@kh.transpose(-1,-2))/math.sqrt(48)).softmax(-1)@vh).transpose(1,2).reshape(B,7,192)[:,:4]   # per-slot attended outputs
    attn_out=F.linear(o,W['core.prelude.0.attn.out.weight'])
    ri=np.array([BR.get(('insulin',0,int(y),k),0.0) for y in s[m]]); rn=np.array([BR.get(('norepinephrine',0,int(y),k),0.0) for y in s[m]])
    capN.append(dict(s=s[m],k=np.full(m.sum(),k),ri=ri,rn=rn,o02=torch.cat([o[...,:48],o[...,96:144]],-1).numpy().copy(),
                     xnorm=X[:,:4].norm(dim=-1).mean(1).numpy().copy(), aonorm=attn_out.norm(dim=-1).mean(1).numpy().copy()))
_=run_rollout_h2(RD,W,[{}],n_paths=1,seed=0,t_hide=1e9,hook_pre=hpre3)
CN={k_:np.concatenate([d[k_] for d in capN]) for k_ in capN[0]}
print('bundles',len(CN['s']),'slot residual norm mean %.1f, attn_out norm mean %.1f'%(CN['xnorm'].mean(),CN['aonorm'].mean()))
isD=(CN['ri']>0)|(CN['rn']>0)
# training rows: drug bundles from all arms (CP, slot0 = u values heads0,2) target rate; none bundles per-slot o (arm0) target 0
XdI=np.hstack([CP['vu'][:,:48],CP['vu'][:,96:144]])[(CP['ri']>0)|(CP['rn']>0)]; ydI=CP['ri'][(CP['ri']>0)|(CP['rn']>0)]
Xn_=CN['o02'][~isD].reshape(-1,96); yn_=np.zeros(len(Xn_))
Xall=np.vstack([XdI,Xn_]); yall=np.r_[ydI,yn_]
wI=np.linalg.solve(Xall.T@Xall+1e-3*np.eye(96),Xall.T@yall)
errD=XdI@wI-ydI; errN=Xn_@wI
print('channel err: drug bundles sd %.3f (ins-only rows sd %.3f), none bundles per-slot |val| mean %.3f max %.3f'%(errD.std(),errD[ydI>0].std(),np.abs(errN).mean(),np.abs(errN).max()))
np.save('/work/wI_chan.npy',wI)
```