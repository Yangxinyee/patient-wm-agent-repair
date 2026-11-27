The rebuilt replica reproduces the new slots poorly: R² −2.9 for the slots and −2.2 for `out`. The stand-in memory, which uses one constant event token for every non-drug bundle, together with the default flags (age embedding not added to keys/values), shifts the operating point too much. It runs fast, 0.01 s for 64 bundles, so many evaluations are cheap.

That doesn't sink the plan. I need the replica's derivative, not its level. The derivative should be dominated by parts that don't touch the memory:
- the prelude FFN and prelude norm (exact from `TOK`);
- the linear state-update path, softplus-step·b_in·E;
- the coda (exact from the new slots).

So the next run will do three things.
1. Write my own prelude that accepts a perturbation added to the slot positions after attention, i.e. exactly where a head-1 value injection lands. I'll check that unperturbed it reproduces the stored prelude outputs `E0`.
2. Compare the replica's fit with `pre=False` and `pre=True`.
3. For 200 drug bundles, compute by central differences how `out` responds to injecting W1o·eⱼ for each of the 48 head-1 value directions, scaled by that bundle's head-1 attention on the action token. I'll do this under both flags and compare.

If the two Jacobians agree closely (median cosine above about 0.9) despite their very different fits, the memory approximation is irrelevant for designing the injection direction. If they disagree, I must rebuild the measurement event tokens first.

```python
import math
Wp0={k:load_original(k).double() for k in NAMES if k.startswith('core.prelude')}
def LNd(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdim=True); v=((x-m)**2).mean(-1,keepdim=True); return (x-m)/torch.sqrt(v+eps)*w+b
def prelude_tail(TOKb, dx=None, Wd=Wp0, ret_att=False):
    p='core.prelude.0.'; x=TOKb.double(); N,T,_=x.shape
    n1=LNd(x,Wd[p+'norm1.weight'],Wd[p+'norm1.bias'])
    q=n1@Wd[p+'attn.q.weight'].T; kv=n1@Wd[p+'attn.kv.weight'].T; k=kv[...,:192]; v=kv[...,192:]
    qh=q.reshape(N,T,4,48).transpose(1,2); kh=k.reshape(N,T,4,48).transpose(1,2); vh=v.reshape(N,T,4,48).transpose(1,2)
    att=torch.softmax(qh@kh.transpose(-1,-2)/math.sqrt(48),-1)
    o=(att@vh).transpose(1,2).reshape(N,T,192)@Wd[p+'attn.out.weight'].T
    x=x+o
    if dx is not None: x=x.clone(); x[:,:4]=x[:,:4]+dx
    n3=LNd(x,Wd[p+'norm3.weight'],Wd[p+'norm3.bias'])
    f=torch.nn.functional.gelu(n3@Wd[p+'ffn.0.weight'].T+Wd[p+'ffn.0.bias'])@Wd[p+'ffn.2.weight'].T+Wd[p+'ffn.2.bias']
    x=x+f; E=LNd(x[:,:4],Wd['core.prelude_norm.weight'],Wd['core.prelude_norm.bias'])
    return (E,att) if ret_att else E
with torch.no_grad():
    Ech,attc=prelude_tail(TOKt,ret_att=True)
    print('prelude_tail vs E0: max abs err %.2e'%float((Ech.float()-E0t).abs().max()), ' head1 slot->u att mean %.4f min %.4f'%(float(attc[:,1,:4,5].mean()),float(attc[:,1,:4,5].min())))
    for pre_ in (False,True):
        hn_,o_=step_full(di[:256], E0t[:256], htft[:256], pre=pre_)
        e1=np.asarray(hn_)-np.asarray(yt[:256]); e2=np.asarray(o_)-np.asarray(ot[:256])
        print(f'pre={pre_}: y rms {np.sqrt((e1**2).mean()):.3f}  out rms {np.sqrt((e2**2).mean()):.4f}  out corr-per-dim median {np.median([np.corrcoef(np.asarray(o_)[:,j],np.asarray(ot[:256])[:,j])[0,1] for j in range(192)]):.3f}')
    W1o_t=torch.as_tensor(wo_o[:,48:96],dtype=torch.float64)
    rng_j=np.random.default_rng(0); sel=np.sort(rng_j.choice(len(di),200,replace=False))
    Tb=TOKt[sel]; Eb=prelude_tail(Tb); a_u=attc[sel][:,1,:4,5]  # (200,4)
    Jv={}
    for pre_ in (False,True):
        eps=0.5; Js=np.zeros((200,192,48))
        for j in range(48):
            dv=W1o_t[:,j]*eps/torch.linalg.norm(W1o_t[:,j])*5  # norm ~2.5
            dx=a_u[:,:,None]*dv[None,None,:]
            Ep=prelude_tail(Tb,dx); Em=prelude_tail(Tb,-dx)
            op=step_full(di[sel],Ep.float(),htft[sel],pre=pre_)[1]; om=step_full(di[sel],Em.float(),htft[sel],pre=pre_)[1]
            Js[:,:,j]=np.asarray(op-om)/(2*float(eps*5/torch.linalg.norm(W1o_t[:,j])))
        Jv[pre_]=Js
    cj=[float((Jv[False][n].ravel()@Jv[True][n].ravel())/np.linalg.norm(Jv[False][n])/np.linalg.norm(Jv[True][n])) for n in range(200)]
    print('Jacobian d out/d a (per unit value coord): pre F vs T cos median %.4f min %.4f; |J| F %.4f T %.4f'%(np.median(cj),np.min(cj),np.linalg.norm(Jv[False],axis=(1,2)).mean(),np.linalg.norm(Jv[True],axis=(1,2)).mean()))
```