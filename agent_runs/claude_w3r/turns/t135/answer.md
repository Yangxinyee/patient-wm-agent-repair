The `Head` class in impl2.py rebuilds the value head from a state: context, z prior, with_code, hidden layer and logits with the transition row. So I can measure prediction changes locally.

Next, at insulin bundles in 15 stays whose next bundle has a glucose_finger reading, I will compute the predicted next-glucose mean three ways: with the true u, and with u replaced by the fixed u of a 2 u/h insulin-only bundle or of a 1 u/h one. These two replacements are in opposite "flip" states. I use z at its prior mean and my local core step.

- **Confirms a harmless rewrite:** the predicted mean moves by well under 1 mg/dL. Then rewriting the encoder output around a fixed vector is safe.
- **Refutes it:** shifts of several mg/dL. Then u's current content matters and must be preserved.

```python
hd=Head({n:load_original(n) for n in NAMES}); hd.W={k:v.double() for k,v in hd.W.items()}
Fnn=torch.nn.functional; ns2['Fnn']=Fnn
def pred_mean_gf(y,o,dt_next,lv,age_s,code_name='glucose_finger'):
    decay=torch.exp(-dt_next*torch.exp(Wt['core.flow.a']))
    ctx_=o+LNt((y*decay).mean(0),'core.out_norm')+mlp_t(torch.tensor([[math.log(dt_next)/10]],dtype=torch.float64),'gap')[0]
    pr=mlp_t(ctx_[None],'z_prior')[0]; zmu=pr[:32]
    view=ctx_+zmu@Wt['z_head.weight'].T+Wt['z_head.bias']; q_=LNt(view+Wt['type_emb.weight'][1],'query_norm')
    c_=CODE[code_name]; num=torch.zeros(8,dtype=torch.float64); lb=fbin(code_name,'value',lv); num[0]=1; num[1]=math.log(1+age_s/60)/10; num[2]=zsc(code_name,'value',lv); num[3]=1
    wc_=q_+Wt['embed.code.weight'][c_]+num@Wt['last_numeric.weight'].T+Wt['last_numeric.bias']+Wt['embed.bins.value.weight'][lb]
    lg=mlp_t(wc_[None],'attr_heads.value_bin')[0]+Wt['embed_transition.value_bin.weight'][c_*17+lb]
    p_=torch.softmax(lg,-1)[1:]; cen_=torch.tensor(VOCAB['numeric'] and [d for d in VOCAB['value_bin_centres'][code_name]],dtype=torch.float64) if isinstance(VOCAB['value_bin_centres'],dict) else None
    return float((p_*cen_).sum()/p_.sum())
fbin=ns_impl['fbin']; zsc=ns_impl['zsc']
u_r2=u_of(ins=2.0); u_r1=u_of(ins=1.0); dif=[]
for s in selS[:15]:
    ev_=recs[recs.stay_id==s].sort_values('hour'); B_=build_bundles(ev_,OPTS); rw=np.where(SST==s)[0]
    if len(B_)!=len(rw): continue
    EU_=[enc_eu(al) for _,al in B_]
    for k in range(1,len(B_)-1):
        if not any(a['code']==CODE['insulin'] and a['kind']==0 for a in B_[k][1]): continue
        if not any(a['code']==CODE['glucose_finger'] for a in B_[k+1][1]): continue
        last=[(t_,a['val'][1]) for t_,al in B_[:k+1] for a in al if a['code']==CODE['glucose_finger']]
        if not last: continue
        tl,lv=last[-1]; age_s=(B_[k][0]-tl)*3600; dtn=(B_[k+1][0]-B_[k][0])*3600
        hp=torch.tensor(SX[rw[k-1],192:960].reshape(4,192),dtype=torch.float64); zk=torch.tensor(SX[rw[k],960:],dtype=torch.float64)
        idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU_[i][0] for i in idx_]); Mage=torch.tensor([(B_[k][0]-B_[i][0])*3600 for i in idx_],dtype=torch.float64)
        dtk=(B_[k][0]-B_[k-1][0])*3600; pm_=[]
        for uu in [EU_[k][1], u_r2, u_r1]:
            y,o,_=core_step_v(hp,dtk,EU_[k][0],uu,zk,Mem,Mage,'std'); pm_.append(pred_mean_gf(y,o,dtn,lv,age_s))
        dif.append(pm_)
dif=np.array(dif); print('n',len(dif),' mean pred (true u) %.1f'%dif[:,0].mean())
print('replace by u(rate2): mean diff %.3f  sd %.3f  max|.| %.3f'%((dif[:,1]-dif[:,0]).mean(),(dif[:,1]-dif[:,0]).std(),np.abs(dif[:,1]-dif[:,0]).max()))
print('replace by u(rate1): mean diff %.3f  sd %.3f  max|.| %.3f'%((dif[:,2]-dif[:,0]).mean(),(dif[:,2]-dif[:,0]).std(),np.abs(dif[:,2]-dif[:,0]).max()))
```