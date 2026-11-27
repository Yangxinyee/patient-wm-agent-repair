Pure float32 binning (rate 7, dose 7) gives an exact match. So the model bins with float32 values against float32 edges, counting edges strictly below. With that, both encoders are exact for every bundle.

Next is the core step. I'll check the z posterior first. For each bundle k ≥ 1 of a few stays I take `out` and h from the tool's state after bundle k−1. I apply the time flow over the gap dt, form ctx = out + out_norm(mean of slots) + gap(log dt / 10), and compare z_post([ctx, e_k, u_k]) with the tool's z.
- **If z matches to about 1e-5:** ctx is right.
- **If not:** the gap scaling or the time units are wrong.

```python
def fbin_f32(code_name, attr, x):
    ed=np.array(NUMV[(code_name,attr)]['fine_bin_edges'],dtype=np.float32); return 1+int(np.sum(ed<np.float32(x)))
def fix_atom2(a):
    a=fix_atom(a)
    if a['act'] and a['kind']!=2:
        nm,r_,d_=raw_drug(a); a['rb']=fbin_f32(nm,'rate',round(r_,6)); a['db']=fbin_f32(nm,'dose',round(d_,6))
    return a
def enc_eu2(atoms):
    A=[fix_atom2(a) for a in atoms]
    ea=[a for a in A if (not a['act']) or a['kind']==2]; ua=[a for a in A if a['act'] and a['kind']!=2]
    e=set_enc(W0,ea,'encode'); u=set_enc(W0,ua,'encode_action') if ua else u_empty
    return e,u
Xs=npz['x'].astype(np.float64); sid_s=npz['stay_id']
row_start={}
for i,s in enumerate(sid_s):
    if s not in row_start: row_start[s]=i
def zcheck(s, opts):
    i0=row_start[s]; B_=BUN[s]; errs=[]
    for k in range(1,len(B_)):
        prev=Xs[i0+k-1]; out_p=torch.tensor(prev[:192]); h_p=torch.tensor(prev[192:960].reshape(4,192))
        dt=(B_[k][0]-B_[k-1][0])*60.0
        e,u=enc_eu2(B_[k][1])
        hf=h_p*torch.exp(-dt*torch.exp(W0['core.flow.a']))
        g=mlp(W0, torch.tensor([math.log(dt)/opts['gs']],dtype=T),'gap')
        ctx=out_p+lnorm(W0,hf.mean(0),'core.out_norm')+g
        z=mlp(W0, torch.cat([ctx,e,u]),'z_post')[:32].detach().numpy()
        errs.append(np.abs(z-Xs[i0+k,960:992]).max())
    return np.array(errs)
import math
for gs in [10.0, 1.0]:
    er=np.concatenate([zcheck(s,{'gs':gs}) for s in stays[:5]])
    print('gap scale',gs,'z err median %.2e max %.2e'%(np.median(er),er.max()))
```