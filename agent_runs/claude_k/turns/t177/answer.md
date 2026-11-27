My rebuilt insulin atom is exactly code[8] + kind[0] + ekind[2] + numeric + dose bin 8 + rate bin 8, with **no categorical rows**. Rows 0 of the categorical tables are zero padding. The real data pipeline very likely adds rows for drug atoms: rate_unit (units/hour), a dose unit (units, mg) in the `unit` table, perhaps action, route or status. A constant missing row, pushed through the encoder MLPs and LayerNorm, can easily produce a rate-dependent error like the one seen.

The cleanest test uses bundles holding only one insulin start atom (no norepinephrine, no END atoms). I'll try all 192 combinations of unit / rate_unit / action / route / status rows for the insulin atom. For each I'll compare both projections, p_ins·u and p_ne·u, with the values implied by the tool.

A correct combination should drop the error to near zero (≲0.05) across all rates. If none does, the bug is in the set encoder itself.

```python
import itertools
e_ins=np.array(VOCAB['numeric'][6]['fine_bin_edges'])
def ins_atom(r_, uu, rr, aa, ro, stt, kind=0):
    z_=np.clip((r_-1.5)/1.0,-10,10); b_=1+int(np.sum(e_ins<r_))
    v_=T['embed.code.weight'][8]+T['embed.kind.weight'][kind]+T['embed.ekind.weight'][2]+num_b+num_W@np.array([0,0,z_,1,z_,1.0])
    v_=v_+T['embed.bins.dose.weight'][b_]+T['embed.bins.rate.weight'][b_]
    v_=v_+T['embed.categorical.unit.weight'][uu]+T['embed.categorical.rate_unit.weight'][rr]+T['embed.categorical.action.weight'][aa]+T['embed.categorical.route.weight'][ro]+T['embed.categorical.status.weight'][stt]
    return v_
sel_=np.nonzero(hi[ok_]&~hn[ok_]&(prev_i==0)&(prev_n==0)&np.isfinite(imp_ins[ok_]))[0]
rates_=cur_i[sel_]; yI=imp_ins[ok_][sel_]; yN=imp_ne[ok_][sel_]
print('n1 bundles',len(sel_),'distinct rates',np.unique(rates_))
# check baseline reproduces drug_atoms
chk_=np.asarray(setenc_my([ins_atom(rates_[0],0,0,0,0,0)],'encode_action'))
V_b=drug_atoms(sidx[int(sid8[ok_[sel_[0]]])], int(np.floor(hr8[ok_[sel_[0]]])), 'both')
print('baseline matches drug_atoms:', len(V_b), np.linalg.norm(chk_-np.asarray(setenc_my(V_b,'encode_action'))))
res_v=[]
ur=np.unique(rates_)
for uu,rr,aa,ro,stt in itertools.product(range(6),range(4),range(2),range(2),range(2)):
    UU={r_:np.asarray(setenc_my([ins_atom(r_,uu,rr,aa,ro,stt)],'encode_action')) for r_ in ur}
    pI=np.array([p_ins_@UU[r_] for r_ in rates_]); pN=np.array([p_ne_@UU[r_] for r_ in rates_])
    res_v.append((float(np.sqrt(np.mean((pI-yI)**2)+np.mean((pN-yN)**2))), float(np.sqrt(np.mean((pI-yI)**2))), float(np.sqrt(np.mean((pN-yN)**2))), (uu,rr,aa,ro,stt)))
res_v.sort(); print('top variants (rmse, rmseI, rmseN, (unit,rate_unit,action,route,status)):'); [print(np.round(x[:3],3), x[3]) for x in res_v[:6]]
print('baseline', [x for x in res_v if x[3]==(0,0,0,0,0)])
```