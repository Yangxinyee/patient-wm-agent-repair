The encoder and z posterior now match the model to float precision (maximum error 6×10⁻⁷ over all 5,245 bundles), and the notes are updated.

Next I implement the core step: time flow, context, z, prelude, the two recurrent layers with slot self-attention and cross-attention to the memory of e vectors, the a_bar update, then c_out and coda. I verify the new h and out against the recorded states, starting from the true previous state. Three details are not fixed by ARCHITECTURE.md, so I test variants:
- whether the memory includes the current bundle's e;
- how the age embedding enters (added to keys after projection, or to memory before projection);
- the age unit (seconds or hours).

An error near 0 for one variant pins down the step. If every variant leaves a sizeable error, a structural assumption is wrong (for example the kv split or the norm placement).

```python
IMPL4 = r'''
def attn_(xq,xkv,W,pre,mask=None,kadd=None,inadd=None):
    q=F.linear(xq,W[pre+'.q.weight'])
    src=xkv if inadd is None else xkv+inadd
    kv=F.linear(src,W[pre+'.kv.weight']); k,v=kv[...,:192],kv[...,192:]
    if kadd is not None: k=k+kadd
    B,Lq,_=q.shape; Lk=k.shape[1]
    q=q.reshape(B,Lq,4,48).transpose(1,2); k=k.reshape(B,Lk,4,48).transpose(1,2); v=v.reshape(B,Lk,4,48).transpose(1,2)
    a=(q@k.transpose(-1,-2))/math.sqrt(48)
    if mask is not None: a=a.masked_fill(~mask[:,None,None,:],float('-inf'))
    o=(a.softmax(-1)@v).transpose(1,2).reshape(B,Lq,192)
    return F.linear(o,W[pre+'.out.weight'])
def layer_sa(x,W,pre):
    xn=LN(x,W,pre+'.norm1'); x=x+attn_(xn,xn,W,pre+'.attn')
    return x+mlp(LN(x,W,pre+'.norm3'),W,pre+'.ffn')
def layer_blk(x,W,pre,M,mask,agee,age_mode):
    xn=LN(x,W,pre+'.norm1'); x=x+attn_(xn,xn,W,pre+'.attn')
    xn=LN(x,W,pre+'.norm2')
    if age_mode=='kadd': x=x+attn_(xn,M,W,pre+'.cross',mask=mask,kadd=agee)
    elif age_mode=='inadd': x=x+attn_(xn,M,W,pre+'.cross',mask=mask,inadd=agee)
    elif age_mode=='kin':
        q_=None; Wk=W[pre+'.cross.kv.weight']
        k=F.linear(M+agee,Wk[:192]); v=F.linear(M,Wk[192:])
        x=x+attn_kv(xn,k,v,W,pre+'.cross',mask)
    return x+mlp(LN(x,W,pre+'.norm3'),W,pre+'.ffn')
def attn_kv(xq,k,v,W,pre,mask):
    q=F.linear(xq,W[pre+'.q.weight']); B,Lq,_=q.shape; Lk=k.shape[1]
    q=q.reshape(B,Lq,4,48).transpose(1,2); k=k.reshape(B,Lk,4,48).transpose(1,2); v=v.reshape(B,Lk,4,48).transpose(1,2)
    a=(q@k.transpose(-1,-2))/math.sqrt(48)
    if mask is not None: a=a.masked_fill(~mask[:,None,None,:],float('-inf'))
    return F.linear((a.softmax(-1)@v).transpose(1,2).reshape(B,Lq,192),W[pre+'.out.weight'])
def core_step(h,out,e,u,dt,M,Mmask,Mage,W,age_mode='kadd',age_unit=1.0,age_div=1.0,z=None):
    hf=h*torch.exp(-dt[:,None,None]*torch.exp(W['core.flow.a']))
    ctx=out+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt)[:,None]/10,W,'gap')
    if z is None: z=mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]
    X=torch.cat([hf+W['core.slots'][None],e[:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    X=layer_sa(X,W,'core.prelude.0'); E=LN(X[:,:4],W,'core.prelude_norm')
    agee=mlp(torch.log1p(Mage/age_unit)[...,None]/age_div,W,'core.age')
    x=hf+E
    for l in ['core.blocks.0.0','core.blocks.0.1']: x=layer_blk(x,W,l,M,Mmask,agee,age_mode)
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    hn=abar*hf+sp*F.linear(E,W['core.b_in.weight'])+(x-(hf+E))
    y=layer_sa(F.linear(hn,W['core.c_out.weight']),W,'core.coda.0')
    return y,LN(y.mean(1),W,'core.out_norm'),z,ctx,hf
'''
open('/work/impl.py','a').write(IMPL4); exec(IMPL4)
# memory for each non-first state row
Eall=E0[rowB]; Uall=U0[rowB]; tS=B0.t.values[rowB]
pos=pd.Series(sidX).groupby(sidX).cumcount().values
start_of=np.zeros(len(sidX),int); 
for i in range(len(sidX)): start_of[i]=i-pos[i]
Lmax=pos.max()+1
def build_mem(idx,incl):
    n=len(idx); M=torch.zeros(n,Lmax,192); mk=torch.zeros(n,Lmax,dtype=torch.bool); ag=torch.zeros(n,Lmax)
    for r,j in enumerate(idx):
        s0=start_of[j]; s1=j+1 if incl else j
        L=s1-s0; M[r,:L]=Eall[s0:s1]; mk[r,:L]=True; ag[r,:L]=torch.as_tensor((tS[j]-tS[s0:s1])*3600.0,dtype=torch.float32)
    return M,mk,ag
sub=nf[:800]
res_v=[]
for incl in [True,False]:
    M,mk,ag=build_mem(sub,incl)
    for am in ['kadd','inadd','kin']:
        for au,adiv in [(1.0,1.0),(3600.0,1.0),(1.0,10.0)]:
            dsub=torch.as_tensor(((mins[sub]-mins[prev[sub]])*60.0).astype(np.float32))
            y,o_,_,_,_=core_step(Hs_[prev[sub]],OUTs[prev[sub]],Eall[sub],Uall[sub],dsub,M,mk,ag,WO,am,au,adiv)
            ey=float((y-Hs_[sub]).abs().mean()); eo=float((o_-OUTs[sub]).abs().mean())
            res_v.append((incl,am,au,adiv,round(ey,5),round(eo,5)))
print(pd.DataFrame(res_v,columns=['incl','age_mode','unit','div','err_h','err_out']).sort_values('err_h').head(8).to_string())
print('scale |h|', float(Hs_[sub].abs().mean()), '|out|', float(OUTs[sub].abs().mean()))
```