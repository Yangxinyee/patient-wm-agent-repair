There are several prepared 300-stay sets (RB, RC, RD, Rs) and 900-stay sets (R0, RT). The local rollout results expose `pm`, `nll`, `seq_si`, `seq_arm` and `drawn`. There are also old with_code arrays (`X0`, `X1`, `Xc`), but I can't recall exactly how they were built. I'll regenerate clean training data with known semantics and keep the code in a file.

The data:
- **Runs:** teacher-forced runs on 300 train stays that are disjoint from my 100-stay validation set `Ro`.
- **Arms (8):** recorded dose plus seven edits: insulin x2, x½, x0, x3, x1.5 from hour 12, and x2 from hours 6 and 16.
- **Saved per glucose/MAP prediction:** the exact head input (`with_code`, using one random prior-z draw) and the target correction r = Σa·d + Σ(c−a)·p(g) computed from that arm's actual doses.

This gives a decoder training set that covers the counterfactual dose regime the evaluation uses.

```python
code_gen = r'''
def gen_wc(Rx, arms, W, a_i, c_i, pf, seed=0, L=4):
    store=[]
    res=run_rollout_h2(Rx,W,arms,n_paths=1,seed=seed,t_hide=1e9,store=store)
    A=Rx['A']; seq_si=res['seq_si']; seq_arm=res['seq_arm']
    DG=dose_stack(Rx,arms,'insulin'); DGn=dose_stack(Rx,arms,'norepinephrine')
    hk=make_lin_oracle(Rx,DGi=DG,a_i=a_i,c_i=c_i,k_i=0.0,pf=pf,t_hide=1e9)
    g=torch.Generator().manual_seed(seed); out={k:[] for k in ['wc','code','arm','si','hj','t','lb','P','pm0','val']}
    for tup in store:
        hn_,hj,base,P,f8,lb_,zmu=tup[:7]
        codes=torch.as_tensor(A.code.values[hj],dtype=torch.long)
        pr=mlp(base,W,'z_prior'); zls=pr[:,32:]
        z=zmu+torch.exp(zls)*torch.randn(zmu.shape,generator=g)
        lg,wc=head_value_logits(base,codes,f8,lb_,torch.zeros(len(hj),dtype=torch.long),W,z,1)
        out['wc'].append(wc.numpy().astype(np.float32)); out['code'].append(A.code.values[hj]); out['arm'].append(seq_arm[hn_])
        out['si'].append(seq_si[hn_]); out['hj'].append(hj); out['t'].append(A.t.values[hj]); out['lb'].append(lb_.numpy())
        out['P'].append(P.numpy().astype(np.float32)); out['val'].append(A.value.values[hj])
    D={k:np.concatenate(v) for k,v in out.items() if len(v)}
    tau=np.round(D['t']).astype(int)
    D['d']=lag_abs(DG,D['arm'],D['si'],tau,8); D['p']=hk.plags(np.zeros(len(tau),int),D['si'],tau,np.zeros((1,1),int),8)
    D['n']=lag_abs(DGn,np.zeros(len(tau),int),D['si'],tau,8)
    return D, res
'''
open('/work/gen_wc.py','w').write(code_gen); exec(code_gen)
print('overlap RD/Ro', len(set(RD['stays'])&set(Ro['stays'])), 'RD in train', np.isin(list(RD['stays']),all_tr).mean())
t0=time.time()
armsW=[{},{'ins':2.0},{'ins':0.5},{'ins':0.0},{'ins':3.0},{'ins':1.5},{'ins':2.0,'ins_from':6.0},{'ins':2.0,'ins_from':16.0}]
Dw,res_w=gen_wc(RD,armsW,W,a_i,c_i,pf_np,seed=0)
print({k:v.shape for k,v in Dw.items()}, round(time.time()-t0,1))
print('codes', {CNAME[c]:int((Dw['code']==c).sum()) for c in np.unique(Dw['code'])})
```