Run 258 confirms the local rollout re-embeds atoms (`arm_embed`) and re-runs both set encoders with whatever weight dict I pass. Edited encoder weights therefore propagate into my local rollouts, and `res['nll']` gives a local teacher-forced factual NLL. So I can build and test the channel entirely locally.

Step 1 is the encoder half. I collect START-atom embeddings for insulin and NE under dose multipliers 0.5–3 applied to all hours. With these I:
- find unused hidden units in `encode_action.mlp1` and `mlp2` that can be repurposed;
- find the quietest directions of the pooled set vector, where the channel can live without disturbing mlp2's other units;
- solve for input weights of four mlp1 units. A linear unit and a constant unit per drug, in GELU's linear regime and switched off for the other drug, whose difference equals the rate.

The design works if the least-squares fit is near-exact (tiny residuals) with a clean on/off gap. A large residual would mean the rate-bin embeddings can't be cancelled linearly.

```python
gelu=lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
ARMS_D=[{'ins':m,'ins_from':-10.0,'ne':m,'ne_from':-10.0} for m in [0.5,1.0,1.5,2.0,3.0]]
A_=Rx['A']; cA_=A_.code.values; kA_=A_.kind.values
isI=(cA_==CODE['insulin'])&(kA_==0); isN=(cA_==CODE['norepinephrine'])&(kA_==0); isU=isI|isN
Wn={k_:(v.numpy() if torch.is_tensor(v) else np.asarray(v)).astype(np.float64) for k_,v in W.items()}
W1,b1,W2_,b2=[Wn['encode_action.mlp1.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
W3,b3,W4,b4=[Wn['encode_action.mlp2.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
XU=[];RU=[]
for a in ARMS_D:
    X=arm_embed(Rx,W,a); X=(X.numpy() if torch.is_tensor(X) else X).astype(np.float64)
    XU.append(X[isU]); RU.append(A_.rate.values[isU]*a['ins'])
XU=np.concatenate(XU); RU=np.concatenate(RU); isIu=np.tile(isI[isU],len(ARMS_D))
H1=X_=None
H1=gelu(XU@W1.T+b1); imp1=H1.std(0)*np.linalg.norm(W2_,axis=0); mx1=np.abs(H1).max(0)
dead1=np.argsort(mx1)[:8]; print('mlp1 least-active units', dead1, np.round(mx1[dead1],4))
# pooled per bundle
key=np.tile(A_.si.values[isU]*1000+A_.bidx.values[isU],len(ARMS_D))+np.repeat(np.arange(len(ARMS_D)),isU.sum())*10**8
uk_,inv=np.unique(key,return_inverse=True); O1=H1@W2_.T+b2
S_=np.zeros((len(uk_),192)); np.add.at(S_,inv,O1); cnt_=np.bincount(inv).astype(float)
Wsz,bsz=Wn['encode_action.size.weight'][:,0],Wn['encode_action.size.bias']
POOL=S_/np.sqrt(cnt_)[:,None]+np.log1p(cnt_)[:,None]*Wsz+bsz
pe=bsz.copy()                                   # empty set pooled
H2=gelu(POOL@W3.T+b3); mx2=np.maximum(np.abs(H2).max(0),np.abs(gelu(pe@W3.T+b3)))
dead2=np.argsort(mx2)[:8]; print('mlp2 least-active units', dead2, np.round(mx2[dead2],4))
Pw=np.vstack([POOL,np.repeat(pe[None],len(POOL)//2,0)]); C_=np.cov(Pw.T); ev_,evec_=np.linalg.eigh(C_)
print('pooled cov smallest eig', np.round(ev_[:6],5), 'largest', np.round(ev_[-3:],2), 'cnt>1 frac', round(float((cnt_>1).mean()),3))
zI=np.clip((RU-1.5)/1.0,-10,10); zN=np.clip((RU-0.46)/0.6,-10,10)
def solve_unit(tgt_on,tgt_off,on_mask,lam=1e-3):
    Xa=np.c_[XU,np.ones(len(XU))]; t=np.where(on_mask,tgt_on,tgt_off)
    sol=np.linalg.solve(Xa.T@Xa+lam*np.eye(193),Xa.T@t); pre=Xa@sol
    return sol, np.abs(pre-t)[on_mask].max(), pre[on_mask].min(), pre[~on_mask].max(), np.linalg.norm(sol[:192])
designs={'I1':(5.5+1.0*zI, 5.5-17+1.0*zN, isIu),'I2':(4.0+0*zI, 4.0-10+0*zN, isIu),
         'N1':(5.5+0.6*zN, 5.5-17+0.6*zI, ~isIu),'N2':(5.0+0*zN, 5.0-10+0*zI, ~isIu)}
SOL={}
for k_,(ton,toff,mk) in designs.items():
    sol,res_on,mn_on,mx_off,nw=solve_unit(ton,toff,mk); SOL[k_]=sol
    print(k_,'max|resid| on %.4f  min pre on %.2f  max pre off %.2f  |w| %.2f'%(res_on,mn_on,mx_off,nw))
```