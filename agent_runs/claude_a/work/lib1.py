
import torch, numpy as np
BLK=['core.blocks.0.0','core.blocks.0.1']
def cl(P0): return {k:(v.clone() if torch.is_tensor(v) else torch.as_tensor(np.asarray(v)).clone()).float() for k,v in P0.items()}
def kill(P0, J, mem=()):
    """Cut h dims J out of the model (no writers, no readers). mem: dims that will carry memory (also cut from ctx->z)."""
    P=cl(P0); J=list(J)
    P['core.slots'][:,J]=0
    pre='core.prelude.0'
    P[pre+'.attn.out.weight'][J,:]=0
    P[pre+'.ffn.2.weight'][J,:]=0; P[pre+'.ffn.2.bias'][J]=0
    P[pre+'.ffn.0.weight'][:,J]=0
    P['core.prelude_norm.weight'][J]=0; P['core.prelude_norm.bias'][J]=0
    for b in BLK:
        for nm in ['attn.out','cross.out']: P[f'{b}.{nm}.weight'][J,:]=0
        P[f'{b}.ffn.2.weight'][J,:]=0; P[f'{b}.ffn.2.bias'][J]=0
        for n in ['norm1','norm2','norm3']: P[f'{b}.{n}.weight'][J]=0
    P['core.b_in.weight'][J,:]=0; P['core.b_in.weight'][:,J]=0
    P['core.c_out.weight'][J,:]=0; P['core.c_out.weight'][:,J]=0
    c='core.coda.0'
    P[c+'.attn.out.weight'][J,:]=0; P[c+'.ffn.2.weight'][J,:]=0; P[c+'.ffn.2.bias'][J]=0
    P[c+'.norm1.weight'][J]=0; P[c+'.norm3.weight'][J]=0
    P['core.out_norm.weight'][J]=0; P['core.out_norm.bias'][J]=0
    P['gap.2.weight'][J,:]=0; P['gap.2.bias'][J]=0
    P['z_head.weight'][J,:]=0; P['z_head.bias'][J]=0
    P['type_emb.weight'][:,J]=0
    P['query_norm.weight'][J]=0
    for m in mem:
        P['z_post.0.weight'][:,m]=0; P['z_prior.0.weight'][:,m]=0
    return P
