
def gen_wc(Rx, arms, W, a_i, c_i, pf, seed=0, L=4):
    store=[]
    res=run_rollout_h2(Rx,W,arms,n_paths=1,seed=seed,t_hide=1e9,want_nll=True,store=store)
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
