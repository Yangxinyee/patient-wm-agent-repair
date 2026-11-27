Found the core parameters: `core.a_in`, `core.delta_in` (so `sp` is presumably softplus(delta_in)), `core.flow.a` and `core.slots`. The code embedding is `embed.code.weight` (13×192), with a `code_bias`.

This run builds stage 3 and verifies stages 1–3 locally:
- per drug, a rank-1 `core.b_in` term reads the E-channel (the intercept is emulated via the LayerNorm-constant direction) and writes into h_mid so that c_out maps it onto a quiet y-direction (`a_y` for insulin, another quiet direction for norepinephrine), 10 per rate unit;
- all edits go into the kernel's `Wt`, and `step_at` measures Δy at action and no-action bundles.

Confirmation would be Δy along the target direction ≈10 × decoded rate, cos ≈1, and ≈0 at no-action bundles.

```python
gpn_=gpt.numpy(); bpn_=bpt.numpy(); tvec=(1/gpn_)/np.sum(bpn_/gpn_); print('sum b/g',round(float(np.sum(bpn_/gpn_)),3))
MY=SX[:,192:960].reshape(-1,4,192).mean(1); evY,QY=np.linalg.eigh(np.cov(MY.T))
ay=a_y/np.linalg.norm(a_y); print('a_y natural sd',round(float((MY@ay).std()),4),'y bottom sd',np.round(np.sqrt(np.maximum(evY[:4],0)),4))
tne=QY[:,1]-(QY[:,1]@ay)*ay; tne/=np.linalg.norm(tne); TY={'ins':ay,'ne':tne}
sp=np.log1p(np.exp(load('core.delta_in').double().numpy())); Cout=load('core.c_out.weight').double().numpy(); Bin=load('core.b_in.weight').double().numpy()
AY=10.; Bin2=Bin.copy()
for nm,dd in DECL.items():
    vr=dd['vE']; vEn=vr[:192]+vr[192]*tvec; a_d=np.linalg.solve(Cout,AY*TY[nm])/sp; Bin2+=np.outer(a_d,vEn); dd['a_d']=a_d; dd['vEn']=vEn
V2n=V2+NF[:, :0].sum() if False else None
evv,evec_=np.linalg.eigh(VV.T@VV/len(VV)); NF5=evec_[:,:5]; W5=RW2[:192,:]
KVn=KV.copy(); KVn[288:336]=V2+NF5@(W5.T-NF5.T@V2); Won=Wo.copy(); Won[:,96:144]=Wo2+(D1-Wo2@NF5)@NF5.T
EDIT={'core.prelude.0.attn.kv.weight':KVn,'core.prelude.0.attn.out.weight':Won,'core.prelude.0.ffn.0.weight':W1n,'core.prelude.0.ffn.0.bias':b1x,'core.prelude.0.ffn.2.weight':W2n,'core.b_in.weight':Bin2}
ORIG={k:Wt[k].clone() for k in EDIT}
def apply_local(on):
    for k,v in EDIT.items(): Wt[k]=torch.tensor(v,dtype=Wt[k].dtype) if on else ORIG[k].clone()
sq=int(META[0][0]); Bq,rq,Eq=stay_data(sq)
apply_local(False); o0=[step_at(Bq,rq,Eq,k) for k in range(len(Bq))]
apply_local(True); o1=[step_at(Bq,rq,Eq,k) for k in range(len(Bq))]; apply_local(False)
print('step_at returns',type(o0[0]).__name__,[getattr(t,'shape',type(t).__name__) for t in (o0[0] if isinstance(o0[0],(tuple,list)) else [o0[0]])])
```