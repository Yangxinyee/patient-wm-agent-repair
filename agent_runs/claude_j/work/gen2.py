
import numpy as np, torch, time
def gen2(s, conds, start_min=720):
    rt=RTs[s]; k_m=first_drug_min(s); k_m=1000000 if k_m is None else k_m
    eps=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(int(s)%100000),dtype=torch.float64)
    Of,Hf,k0=run_cf(s,{},k_m); out={'f':heads_full(Of,Hf,rt,eps)}
    for (dr,m) in conds:
        Oc,Hc,k1=run_cf(s,{dr:m},start_min); out[(dr,m)]=heads_full(Oc,Hc,rt,eps)
    return out
def mean_of(lg,ev):
    p=torch.softmax(torch.as_tensor(lg),-1).mean(1).numpy()
    return np.array([p[i,1:]@CEN[ev[i]] for i in range(len(ev))])
