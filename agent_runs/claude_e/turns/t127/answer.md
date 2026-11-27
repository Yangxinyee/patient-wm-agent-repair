That run showed the edit behaves as designed. With `make_edit(1,1)`, the finger-glucose prediction moves by about +1.1 to +1.4 mg/dL per u/h of lag-1 insulin deviation. The arterial MAP prediction moves by about +1.0 mmHg per unit of norepinephrine-rate deviation. On top of that is readout noise: SD about 1.8 for finger glucose and 0.7 for arterial MAP. The NLL cost stays at a few thousandths per reading.

So a_ins is close to "mg/dL per u/h of deviation" and a_ne to "mmHg per mcg/kg/min of deviation". To get the lag-1 insulin correction of about −2.25 from my one-step analysis, a_ins should be about −1.9. For norepinephrine I will start at a_ne = +3, the size of the lag-1 response I estimated. I will calibrate it on a rollout.

Next I will:
- build this candidate;
- estimate its factual NLL change offline on the five training batches;
- find the model tensor names for the four head tensors by exact match with the originals;
- save the edit, so the next step can be a dose_edit "both" request on the tuning stays.

I expect the rollout effect to grow beyond the one-step value through the model's own glucose persistence. If the edited insulin ×2 readout lands near my truth estimate of −13 to −14, Plan A is supported. If it stays near the one-step size, it is not.

```python
cand=dict(a_ins=-1.9,a_ne=3.0)
E=make_edit(cand['a_ins'],cand['a_ne'])
tn=0; td=0
for i in range(5):
    Xs_,R_=datasets[i]; o=head_run(Xs_,R_,E); pmb,nllb,_,_=base_out[i]
    dn=npv(o[1])-nllb; tn+=len(dn); td+=dn.sum()
    print(i,len(dn),'dNLL %.4f'%dn.mean(),{c:round(float(dn[R_['code'].values==c].mean()),4) for c in [6,10,7,5]})
print('train overall dNLL per reading %.4f'%(td/tn))
names_map={}
for k,v in ORIG.items():
    m=[n for n in NAMES if tuple(SHAPES[n])==tuple(v.shape) and torch.allclose(torch.as_tensor(load_original(n)).float(),v.float(),atol=1e-6)]
    names_map[k]=m; print(k,m)
if all(len(v)==1 for v in names_map.values()):
    for k,v in E.items(): save_edit(names_map[k][0], v.float())
    print('saved',edited())
    for k in E: print(k,'roundtrip maxdiff',float((torch.as_tensor(load(names_map[k][0])).float()-E[k].float()).abs().max()))
```