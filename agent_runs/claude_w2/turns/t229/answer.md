Run 223 gives the exact core step, which settles where a dose signal can enter. The current action's encoding u (from its own `encode_action` weights, so editing it would not touch measurements) reaches the state through one place only: the self-attention of `core.prelude.0`, where the 4 slot tokens read the 7 tokens [slots, e, u, z_in]. After that it passes linearly through b_in (h update) and c_out, then the coda layer and two LayerNorms to `out`. The decaying h is the only carrier, and the memory never sees doses.

So a clean channel has to be written at that attention step. The design is a rank-1 addition to the attention out-projection of the head that reads u: residual += c·(w48·head_output)·vdir. This was the `make_W_chan` idea. It only works if some head attends to u with a stable weight and its value of u is close to linear in the rate.

Next I capture the prelude inputs at every bundle in a factual teacher-forced pass on 300 of the decoder stays. For each head I measure, at insulin and norepinephrine bundles:
- the slot→u attention weight (mean and sd);
- how well a linear readout of the head's slot-mean output predicts the rate (R²).

**Confirm:** a head with u-attention ≳ 0.3, a small spread, and rate R² ≳ 0.9. **Refute:** if no head carries the rate cleanly, I will inject the dose through `encode_action`/z instead.

```python
print({k:SHAPES[k] for k in NAMES if k.startswith('core.prelude.0')})
cap=[]
Ai=RT['A']; isS=(Ai.kind.values==0)
rateI={};rateN={}
for (si,k),idx in RT['atoms_sk'].items():
    for j in idx:
        if isS[j] and Ai.code.values[j]==CODE['insulin']: rateI[(si,k)]=float(Ai.rate.values[j])
        if isS[j] and Ai.code.values[j]==CODE['norepinephrine']: rateN[(si,k)]=float(Ai.rate.values[j])
def hcap(k,act_n,hf,ctx,e,u,z,c):
    s=c['seq_si'][act_n]; ri=np.array([rateI.get((int(x),k),0.0) for x in s]); rn=np.array([rateN.get((int(x),k),0.0) for x in s])
    m=(ri>0)|(rn>0)|(np.random.rand(len(s))<0.05)
    if not m.any(): return
    mi=torch.as_tensor(np.where(m)[0])
    X=torch.cat([hf[mi]+W['core.slots'][None],e[mi][:,None],u[mi][:,None],F.linear(z[mi],W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    cap.append(dict(X=X.clone(),ri=ri[m],rn=rn[m]))
sub_ids=list(RT['stays'][:300]); Rs=prep_rollout(tr,sub_ids)
rateI={};rateN={}
As=Rs['A']; isS=(As.kind.values==0)
for (si,k),idx in Rs['atoms_sk'].items():
    for j in idx:
        if isS[j] and As.code.values[j]==CODE['insulin']: rateI[(si,k)]=float(As.rate.values[j])
        if isS[j] and As.code.values[j]==CODE['norepinephrine']: rateN[(si,k)]=float(As.rate.values[j])
_=run_rollout_h2(Rs,W,[{}],n_paths=1,seed=0,t_hide=1e9,hook_pre=hcap)
X7=torch.cat([c_['X'] for c_ in cap]); RI=np.concatenate([c_['ri'] for c_ in cap]); RN=np.concatenate([c_['rn'] for c_ in cap])
xn=LN(X7,W,'core.prelude.0.norm1'); q=F.linear(xn,W['core.prelude.0.attn.q.weight']); kv=F.linear(xn,W['core.prelude.0.attn.kv.weight'])
B_=len(X7); qh=q.reshape(B_,7,4,48).transpose(1,2); kvh=kv.reshape(B_,7,2,4,48); kh=kvh[:,:,0].transpose(1,2); vh=kvh[:,:,1].transpose(1,2)
att=torch.softmax(qh@kh.transpose(-1,-2)/math.sqrt(48),-1)   # B,4heads,7q,7k
o=att@vh   # B,4,7,48
print('rows',B_,'ins',int((RI>0).sum()),'ne',int((RN>0).sum()))
for hd in range(4):
    au_i=att[RI>0][:,hd,:4,5].mean(-1); au_n=att[RN>0][:,hd,:4,5].mean(-1)
    os_=o[:,hd,:4].mean(1).numpy()
    def r2(mask,y):
        Xr=np.c_[np.ones(mask.sum()),os_[mask]]; b,*_=np.linalg.lstsq(Xr,y[mask],rcond=None); return 1-np.var(y[mask]-Xr@b)/np.var(y[mask])
    mI=RI>0; mN=RN>0
    print('head',hd,'att->u ins %.3f+-%.3f ne %.3f+-%.3f | R2 rate ins %.3f ne %.3f'%(au_i.mean(),au_i.std(),au_n.mean(),au_n.std(),r2(mI,RI),r2(mN,RN)))
```