
def collect_states_ins(R_, arms, W, idsel=None):
    A_=R_['A']; codeA=A_.code.values; kindA=A_.kind.values
    insS=np.where((codeA==CODE['insulin'])&(kindA==0))[0]
    sk2j={(int(A_.si.values[j]),int(A_.bidx.values[j])):j for j in insS}
    store=[]
    def hs(k,act_n,h,out,ctx):
        s=ctx['seq_si'][act_n]; m=np.array([(int(x),k) in sk2j for x in s])
        if not m.any(): return
        mi=np.where(m)[0]; n=act_n[mi]
        js=np.array([sk2j[(int(x),k)] for x in s[mi]])
        store.append(dict(n=n,j=js,arm=ctx['seq_arm'][n],si=s[mi],u=ctx['u'][mi].clone(),Ep=ctx['Ep'][mi].mean(1).clone(),
                          hn=ctx['hn'][mi].reshape(len(mi),-1).clone(),y=h[n].reshape(len(mi),-1).clone(),out=out[n].clone()))
    res_=run_rollout_h(R_,W,arms,n_paths=1,seed=0,t_hide=1e9,hook_state=hs)
    C={k_:(np.concatenate([d[k_] for d in store]) if not torch.is_tensor(store[0][k_]) else torch.cat([d[k_] for d in store]).numpy()) for k_ in store[0]}
    return C
