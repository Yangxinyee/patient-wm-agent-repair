The earlier impulse calibration used a compensated prelude rather than the original one. I’ll verify its gains against the original update and check that the paired features reproduce the saved arrays. A large calibration discrepancy would require new directions before fitting intervention targets.

```python
print("VOCAB CODES",VOCAB['codes'])
orig_overrides228={n:load_original(n) for n in zero169}
def response_original228(injections):
    inj=tensor153(injections); m=len(inj)
    tok=tensor153(tokens_orig169).repeat(m,1,1)
    ep=prelude155(tok,overrides=orig_overrides228,
                  injection=inj[:,None,:].repeat_interleave(n186,dim=0))[0]
    hp,op=state_update82(hf169.repeat(m,1,1),ep,mem169.repeat(m,1,1),
                          mask169.repeat(m,1),age_emb169.repeat(m,1,1),0)
    return arr183(phys171(view171(hp,op,dt169.repeat(m)))).reshape(m,n186,4).mean(1)
T4_228=np.array([[-1.94,0],[-1.94,0],[0,3.2],[0,3.2]],dtype=np.float64)
dirs_full228=J186.T@np.linalg.solve(J186@J186.T+1e-10*np.eye(4),T4_228)
print("FULL-SENSOR DIRECTION NORMS",np.linalg.norm(dirs_full228,axis=0).round(3))
ins228=np.stack([np.zeros(192)]+[D[:,j]*s for D in [directions187,dirs_full228]
                               for j in range(2) for s in [-1,1]])
rr228=response_original228(ins228)
print("ORIGINAL-UPDATE IMPULSE RESPONSES")
for k,(label,j,s) in enumerate([(label,j,s) for label in ['primary','full_sensor'] 
                              for j in range(2) for s in [-1,1]],start=1):
    print(label,j,s,np.round(rr228[k]-rr228[0],5).tolist())
base_override228={n:tensor153(x) for n,x in checkpoint222.items()}
base_override228['core.prelude.0.ffn.2.weight']=w2_220
base_override228['core.prelude.0.ffn.2.bias']=b2_220
def raw_hidden228(tokens):
    parts=prelude155(tokens,overrides=base_override228)
    aft=parts[1]
    hh=F.gelu(F.linear(layer_norm(aft,'core.prelude.0.norm3'),w0_220,b0_220))
    raw=aft+F.linear(hh,w2_220,b2_220)
    return arr183(hh).reshape(-1,768),arr183(raw).reshape(-1,192),parts
hbase228,rbase228,partsbase228=raw_hidden228(tokens206)
tokhalf228=tensor153(tokens206).clone()
tokhalf228[:,5]=changed227
tokhalf228[:,6]=F.linear(posterior153(C154,E154,changed227,post206),
                         tensor153(zin154),tensor153(zb154))
hhalf228,rhalf228,partshalf228=raw_hidden228(tokhalf228)
dxhalf228=np.column_stack(((hhalf228-hbase228)/hs220,np.zeros(len(hbase228))))
print("ARRAY CHECKS",
      'H baseline RMS',float(np.sqrt(np.mean((hbase228-H220)**2))),
      'raw baseline RMS',float(np.sqrt(np.mean((rbase228-arr183(current_raw220).reshape(-1,192))**2))),
      'contrast feature RMS',float(np.sqrt(np.mean((dxhalf228-contrast_features220[(0,.5)])**2))))
print("SPLIT COUNTS",tr220.dtype,int(np.count_nonzero(tr220)),int(np.count_nonzero(va220)))
```
