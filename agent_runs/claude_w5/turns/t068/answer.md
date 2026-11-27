No time-gap convention gets close to zero; the best is still 0.14. So the main error is elsewhere, probably in the bundle encoding e_k. Even single-reading bundles are off.

Next I'll ablate e_k: set it to zero, and drop in turn the `kq`, unit, bin and numeric terms. I'll also try LayerNorm on each atom vector. The aim is to see which parts the saved z responds to. If removing a term lowers the error, that term is being added wrongly.

```python
def atom_vec3(a_,opt):
    c_=meta[a_['code']]; v=Wt['embed.code.weight'][c_['id']]+Wt['embed.kind.weight'][a_['kind']]+Wt['embed.ekind.weight'][ekid[c_['event_kind']]]
    if 'kq' in opt: v=v+Wt['embed.kq.weight'][0]
    nv=[0.]*6
    if 'value' in a_:
        nv[0]=zsc(a_['code'],'value',a_['value']); nv[1]=1.
        if 'bins' in opt: v=v+Wt['embed.bins.value.weight'][fbin(a_['code'],'value',a_['value'])]
    if c_['unit'] is not None and 'unit' in opt: v=v+Wt['embed.categorical.unit.weight'][U2[c_['unit']]]
    if 'num' in opt: v=v+Wt['embed.numeric.weight']@torch.tensor(nv)
    v=v+Wt['embed.numeric.bias']
    if 'ln' in opt: v=Fn.layer_norm(v,(192,))
    return v
base={'kq','unit','bins','num'}
for nm_o,opt in [('full',base),('-kq',base-{'kq'}),('-unit',base-{'unit'}),('-bins',base-{'bins'}),('-num',base-{'num'}),('+ln',base|{'ln'}),('e=0',None)]:
    E_=[]
    for (p_,c_,dt,_),(sidx,bi) in zip(cases,[(s__,b__) for s__ in list(bundles_B.keys())[:3] for b__ in range(1,30) if all(a_['code'] not in ('insulin','norepinephrine','icu_admission') for a_ in bundles_B[s__][b__][1])]):
        out_=X0[p_,:192]; hf=X0[p_,192:960].reshape(4,192)*torch.exp(-dt*flow)
        e_=torch.zeros(192) if opt is None else enc(torch.stack([atom_vec3(a_,opt) for a_ in bundles_B[sidx][bi][1]]),'encode')
        E_.append((mlp(torch.cat([out_+lnp(hf.mean(0),'core.out_norm'),e_,torch.zeros(192)]),'z_post')[:32]-X0[c_,960:992]).numpy())
    print(nm_o, round(float(np.sqrt((np.array(E_)**2).mean())),4))
```