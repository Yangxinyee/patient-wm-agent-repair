Now z, h and out all match the tool exactly: `step_local(k)` reproduces the model's state update from the previous tool state. So I can measure leverage locally.

The experiment uses 40 drug-A bundles that are followed directly by a glucose_finger reading. At each one I push the z token by 0.3 along each of the 192 basis directions and record the change in the head's predicted mean for the next glucose_finger reading. As a side effect I also record the change for map_arterial. This gives a per-bundle sensitivity matrix.

The hijack plan works if one token direction moves the next glucose prediction consistently across bundles, at least a few units per unit of push, while MAP barely moves. If the effects are small or inconsistent in sign, I will inject further downstream instead.

```python
t0=time.time()
codes4=('glucose_finger','map_arterial'); LVr={c:np.full(N3,np.nan) for c in codes4}; LTr={c:np.full(N3,np.nan) for c in codes4}; hasr={c:np.zeros(N3,bool) for c in codes4}
for c in codes4:
    v=np.nan; t_=np.nan
    for k in range(N3):
        if first[k]: v=np.nan; t_=np.nan
        LVr[c][k]=v; LTr[c][k]=t_
        for a in atoms_by_bundle[(int(sid[k]),int(round(hr[k]*60)))]:
            if a['code']==c and 'value' in a: v=a['value']; t_=tm[k]; hasr[c][k]=True
hasI=S['has_insulin']; hasN=S['has_norepinephrine']
kA=[k for k in range(N3-1) if hasI[k] and not first[k+1] and hasr['glucose_finger'][k+1]]
rngA=np.random.default_rng(1); kAs=rngA.choice(kA,40,replace=False); eps=0.3
CT=[]; meta=[]
for k in kAs:
    k=int(k); k1=k+1; dt1=np.array([dtl[k1]])
    for d in range(-1,192):
        add=None
        if d>=0: add=torch.zeros(192); add[d]=eps
        _,_,h,o=step_local(k,ztok_add=add)
        CT.append(ctx_of_state(o.double().reshape(1,192),h.double().reshape(1,4,192),dt1).reshape(-1))
    meta.append(k1)
CT=torch.stack(CT); n_=len(kAs); rep=lambda a: np.repeat(a,193)
k1s=np.array(meta)
ag=lambda c: (tm[k1s]-LTr[c][k1s])*60.0
mg,_=value_mean(CT,['glucose_finger']*len(CT),rep(LVr['glucose_finger'][k1s]),rep(ag('glucose_finger')))
mm,_=value_mean(CT,['map_arterial']*len(CT),rep(LVr['map_arterial'][k1s]),rep(ag('map_arterial')))
mg=mg.numpy().reshape(n_,193); mm=mm.numpy().reshape(n_,193)
G=(mg[:,1:]-mg[:,:1])/eps; Gm=(mm[:,1:]-mm[:,:1])/eps; np.save('/work/G_tok_A.npy',np.stack([G,Gm]))
gb=G.mean(0); v=gb/np.linalg.norm(gb); s_=G@v; sm=Gm@v
print('time',round(time.time()-t0,1),' |gbar|',round(float(np.linalg.norm(gb)),3),' per-bundle effect along gbar: mean',round(float(s_.mean()),3),'sd',round(float(s_.std()),3),' frac same sign',round(float((s_>0).mean()),2))
print('MAP effect along gbar mean/sd',round(float(sm.mean()),3),round(float(sm.std()),3),' max single-dir |G| mean',round(float(np.abs(G).mean()),3))
```