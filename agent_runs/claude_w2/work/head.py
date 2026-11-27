
import numpy as np, torch, torch.nn.functional as F, math
def CEN_of(code_ids):
    cm=np.zeros((13,16))
    for nm,arr in V['value_bin_centres'].items(): cm[CODE[nm]]=np.array(arr)
    return torch.as_tensor(cm[np.asarray(code_ids)],dtype=torch.float32)
def head_base(out_p,h_p,dt_s,W,gdiv=10.0):
    hf=h_p*torch.exp(-dt_s[:,None,None]*torch.exp(W['core.flow.a']))
    return out_p+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt_s)[:,None]/gdiv,W,'gap')
def head_value_logits(base,codes,f8,lastbin,status,W,z,ty=3):
    view=base+F.linear(z,W['z_head.weight'],W['z_head.bias'])
    q=LN(view+W['type_emb.weight'][ty],W,'query_norm')
    wc=(q+W['embed.code.weight'][codes]+F.linear(f8,W['last_numeric.weight'],W['last_numeric.bias'])
        +W['embed.bins.value.weight'][lastbin]+W['embed.bins.rate.weight'][0]+W['last_status.weight'][status])
    return mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes*17+lastbin], wc
def head_probs_mc(base,codes,f8,lastbin,status,W,ty=3,nz=0,seed=0):
    pr=mlp(base,W,'z_prior'); zmu,zls=pr[:,:32],pr[:,32:]
    if nz==0:
        lg,_=head_value_logits(base,codes,f8,lastbin,status,W,zmu,ty); return lg.softmax(-1)
    g=torch.Generator().manual_seed(seed); P=0
    for i in range(nz):
        z=zmu+torch.exp(zls)*torch.randn(zmu.shape,generator=g)
        lg,_=head_value_logits(base,codes,f8,lastbin,status,W,z,ty); P=P+lg.softmax(-1)
    return P/nz
