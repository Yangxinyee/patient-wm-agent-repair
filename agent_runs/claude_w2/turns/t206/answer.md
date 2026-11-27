Prelude heads 0 and 2 put their full attention (weight 1.00, sd 0) on the u token at every insulin bundle. Heads 1 and 3 put 0.8–0.9 on it, and the z token gets nothing. So each slot receives that head's value vector of u without dilution. The dose information that is precise in u (rate R² 0.93) is copied cleanly into the state, and the blurring happens later, in the core's downstream processing.

This suggests an edit that removes nothing. I would add a rank-1 term to the output projection of head 0 (and/or head 2). It decodes the insulin rate from that head's 48-dimension value vector of u and writes it into one chosen slot direction. That gives the state a clean rate channel. The head then reads it, subtracts the policy dose, which it can infer from the last-glucose register and the state, and shifts the glucose prediction accordingly.

Two things must hold first:
1. **Precise decoding.** The insulin rate must be decodable from those head values: R² well above 0.95 (with a small penalty), about 0 for empty and norepinephrine-only u, and slopes of about 1 under the dose arms.
2. **Known non-insulin attention.** I need to know what heads 0 and 2 attend to at bundles without insulin, so the channel stays quiet there.

This run measures both, at all bundle types and under the dose arms.

```python
capP=[]
A_=RD['A']; cA=A_.code.values; kA=A_.kind.values
def bundle_rates(R_,arms_):
    out_={}
    for drug in ['insulin','norepinephrine']:
        js=np.where((cA==CODE[drug])&(kA==0))[0]
        for ai,a in enumerate(arms_):
            mlt=arm_mult(R_,a,drug); 
            for j in js: out_[(drug,ai,int(A_.si.values[j]),int(A_.bidx.values[j]))]=A_.rate.values[j]*mlt[j]
    return out_
BR=bundle_rates(RD,armsD[:6])
def hpre2(k,act_n,hf,ctx,e,u,z,c):
    s=c['seq_si'][act_n]; ar=c['seq_arm'][act_n]
    X=torch.cat([hf+W['core.slots'][None],e[:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    xn=LN(X,W,'core.prelude.0.norm1'); q=F.linear(xn,W['core.prelude.0.attn.q.weight']); kv=F.linear(xn,W['core.prelude.0.attn.kv.weight'])
    B=X.shape[0]; qh=q.reshape(B,7,4,48).transpose(1,2); kh=kv[...,:192].reshape(B,7,4,48).transpose(1,2)
    a=((qh@kh.transpose(-1,-2))/math.sqrt(48)).softmax(-1)
    vu=kv[:,5,192:]   # value vector of u token (4 heads x 48)
    ri=np.array([BR.get(('insulin',int(x),int(y),k),0.0) for x,y in zip(ar,s)]); rn=np.array([BR.get(('norepinephrine',int(x),int(y),k),0.0) for x,y in zip(ar,s)])
    capP.append(dict(s=s,ar=ar,k=np.full(len(s),k),ri=ri,rn=rn,vu=vu.numpy().copy(),xu=xn[:,5].numpy().copy(),att=a[:,:,:4,:].mean(2).numpy().copy()))
_=run_rollout_h2(RD,W,armsD[:6],n_paths=1,seed=0,t_hide=1e9,hook_pre=hpre2)
CP={k_:np.concatenate([d[k_] for d in capP]) for k_ in capP[0]}
hasI=CP['ri']>0; hasN=CP['rn']>0
# u token empty? detect by bundle having no drug
typ=np.where(hasI&hasN,'both',np.where(hasI,'ins',np.where(hasN,'ne','none')))
for t_ in ['ins','ne','both','none']:
    m=typ==t_; print(t_, m.sum(), 'mean attn (heads0..3) to [s*,e,u,z]:', [np.round([CP['att'][m,h,:4].sum(1).mean(),CP['att'][m,h,4].mean(),CP['att'][m,h,5].mean(),CP['att'][m,h,6].mean()],2).tolist() for h in [0,2]])
# decodability of insulin rate (all bundle types where head attends u) from value vectors
grpP=CP['s']; armP=CP['ar']
def cvP(X,y,alpha):
    pr=np.zeros(len(y))
    for trn_,tst_ in GroupKFold(5).split(X,y,grpP):
        m_=Ridge(alpha=alpha).fit(X[trn_],y[trn_]); pr[tst_]=m_.predict(X[tst_])
    return pr
for nm_,X in [('xu(LN1 u)',CP['xu']),('v head0',CP['vu'][:,:48]),('v head2',CP['vu'][:,96:144]),('v head0+2',np.hstack([CP['vu'][:,:48],CP['vu'][:,96:144]])),('v all',CP['vu'])]:
    for al in [1e-4,1e-2]:
        pr=cvP(X,CP['ri'],al); res_=pr-CP['ri']
        f_=armP==0
        print(nm_,al,'R2 all-bundles fact %.4f'%(1-np.mean(res_[f_]**2)/np.var(CP['ri'][f_])),'err sd ins-bundles fact %.3f'%res_[f_&hasI].std(),
              'mean|err| none %.3f ne %.3f'%(np.abs(res_[f_&(typ=="none")]).mean(),np.abs(res_[f_&(typ=="ne")]).mean()),
              'slope by arm', [round(float(np.polyfit(CP['ri'][(armP==a)&hasI],pr[(armP==a)&hasI],1)[0]),3) for a in range(6)])
```