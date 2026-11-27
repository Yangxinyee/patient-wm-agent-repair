I now have the exact attention code: scores q·k/√48, with keys and values from `attn.kv` (first 192 rows keys, last 192 values), computed from the LayerNormed tokens.

Next I'll measure the attention scores of the four slot queries in each of the four heads over the 7 prelude tokens (4 slots, e, u, z). This uses teacher-forced states from 20 tuning stays, split by drug versus non-drug bundles. I'll report:
- which token wins each head;
- the gap from the winner to the runner-up;
- how negative a fixed "dummy" key can make u's score.

The dummy is the LayerNormed shift b′ that would retire u in the swap design. I'll also store, per query, the vectors that map a token's normalised encoding to its score, so b′ can be optimised next.

**Feasible** if one direction pushes u's score at least about 10 below the winning token for every head and query. **Infeasible** if the queries are too diverse for a single constant.

```python
Wq=W0['core.prelude.0.attn.q.weight']; Wkv=W0['core.prelude.0.attn.kv.weight']
def collect_att(W, B_):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; rec=[]
    for k_ in range(1,len(B_)):
        t=times[k_]; dt=(t-times[k_-1])*3600.0; ea,ua=split_fix(B_[k_][1])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        hf=h*torch.exp(-dt*torch.exp(W['core.flow.a'])); g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
        ctx=out+bln(W,hf.mean(1),'core.out_norm')+g
        z=bmlp(W,torch.cat([ctx,e[None],u[None]],-1),'z_post')[:,:32]; zt=blin(W,z,'z_in')
        tok=torch.cat([hf+W['core.slots'],e[None,None],u[None,None],zt[:,None]],1)
        a1=bln(W,tok,'core.prelude.0.norm1'); q=(a1@Wq.T)[0,:4].reshape(4,4,48); kk=(a1@Wkv.T)[0,:,:192].reshape(7,4,48)
        lg=torch.einsum('ihd,jhd->hij',q,kk)/math.sqrt(48)
        A_=torch.einsum('hdk,ihd->hik',Wkv[:192].reshape(4,48,192),q)/math.sqrt(48)
        rec.append((len(ua)>0, lg.numpy(), A_.numpy()))
        out,h,_=bstep_z(W,out,h,dt,e[None],u[None],Mt,ages)
    return rec
W0z=dict(W0); W0z['z_head.weight']=torch.zeros_like(W0['z_head.weight'])
AR=[]
for s in keys[:20]: AR+=collect_att(W0z,BUN[s])
isd=np.array([r[0] for r in AR]); LG=np.array([r[1] for r in AR]); AA=np.array([r[2] for r in AR])
print('bundles',len(AR),'drug',isd.sum())
tn=['s0','s1','s2','s3','e','u','z']
for typ,m in [('drug',isd),('nondrug',~isd)]:
    L_=LG[m]; win=L_.argmax(-1); srt=np.sort(L_,-1); gap=srt[...,-1]-srt[...,-2]
    for hh in range(4):
        wc=np.bincount(win[:,hh].ravel(),minlength=7)
        print(typ,'head',hh,'winner counts',dict((tn[j],int(wc[j])) for j in range(7) if wc[j]),'gap p5 %.1f med %.1f'%(np.quantile(gap[:,hh],.05),np.median(gap[:,hh])),'u-logit minus top med %.1f'%np.median(L_[:,hh,:,5]-srt[:,hh,:,-1]))
```