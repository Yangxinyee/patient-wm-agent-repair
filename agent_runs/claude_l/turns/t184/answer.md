`z_in.weight` is not zero (norm 8.1), so the z token is ignored for some other reason. The likely cause is that the prelude's attention gives it essentially zero weight. The h test also shows something important: a perturbation written into h at a drug bundle moves glucose at lag 1 (2.2) but is nearly gone by lag 2 (14%) and lag 4 (1%). The network's own dynamics forget within about an hour, so any multi-hour dose memory would have to be re-written at every step. Next I'll print the prelude attention weights from the four slots to the e, u and z tokens, per head, for typical reading and drug bundles. If the weight on the z token is effectively 0 because its key scores are very negative, that explains the dead channel and tells me what a z-token edit would need to change.

```python
def prelude_att(W, s, k):
    B_=BUN[s]; i0=row_start[s]; prev=torch.tensor(Xs[i0+k-1],dtype=T)
    dt=(B_[k][0]-B_[k-1][0])*3600.0; out_p=prev[:192][None]; h_p=prev[192:960].reshape(1,4,192)
    hf=h_p*torch.exp(-dt*torch.exp(W['core.flow.a'])); ctx=out_p+bln(W,hf.mean(1),'core.out_norm')+bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    ea,ua=split_fix(B_[k][1]); e=(set_enc(W,ea,'encode') if ea else set_enc(W,[],'encode'))[None]; u=(set_enc(W,ua,'encode_action') if ua else u_empty)[None]
    z=bmlp(W,torch.cat([ctx,e,u],-1),'z_post')[:,:32]
    tok=torch.cat([hf+W['core.slots'],e[:,None],u[:,None],blin(W,z,'z_in')[:,None]],1)
    a=bln(W,tok,'core.prelude.0.norm1'); L='core.prelude.0.attn'
    q=(a@W[L+'.q.weight'].T).reshape(1,7,4,48).transpose(1,2); kv=a@W[L+'.kv.weight'].T; kk=kv[...,:192].reshape(1,7,4,48).transpose(1,2)
    att=torch.softmax((q@kk.transpose(-1,-2))/math.sqrt(48),-1)[0]   # (H,7,7)
    return att[:, :4, :]   # slots -> all tokens
for s in ks[:2]:
    B_=BUN[s]
    for k in [10, 11, 12]:
        at=prelude_att(W0,s,k)
        print(s, k, B_[k][0], 'slot->[e,u,z] per head (mean over slots):', np.round(at[:,:,4:].mean(1).numpy(),4).tolist())
```