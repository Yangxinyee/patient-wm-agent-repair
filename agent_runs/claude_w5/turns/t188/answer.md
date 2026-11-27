Two findings:
- **Predicted mean.** It is Σ p(bin)·centre over bins 1–16. My 4-draw local means match the stored targets to within z-draw noise (rmse 2.05).
- **Belief directions.** The h_new change from a lower glucose (higher MAP) reading is identical across the four slots. Across bundles it is moderately consistent: median cosine to the mean direction 0.68 for glucose and 0.82 for MAP.

So one write direction shared by all slots is structurally right. The open question is whether the fixed mean direction moves the next predicted reading as much as a real reading change does.

Next, for each drug bundle followed by a glucose finger (MAP arterial) reading, I will compare the next predicted mean under three states, all with common random draws:
1. the original state;
2. the state after the h:00 reading was perturbed (state only);
3. the original step with the fixed mean direction injected into h_new.

If the injection's mean shift is close to the reading-perturbation shift and correlates with it, the fixed direction is adequate.

```python
RK={s:run_tf(bundles_B[s],keep=True) for s in sids}
def mhead(o,yy,dt,i,nz=4):
    v=view0(o,yy,dt)[None]; P_=probs_from_v0(v,fixed[i:i+1],TRr[i:i+1],nz=nz,seed=0)
    return float((P_[0,1:]*cen[i]).sum())
mvg=torch.stack([d['vg'] for d in D if 'vg' in d]).double().mean(0)
mvm=torch.stack([d['vm'] for d in D if 'vm' in d]).double().mean(0)
out={}
for nm,code,mv,amp in [('vg','glucose_finger',mvg,10.0),('vm','map_arterial',mvm,5.0)]:
    a=[]
    for d in D:
        if nm not in d: continue
        s,k=d['sid'],d['k']; BB=bundles_B[s]
        if k+1>=len(BB) or (s,k+1,code) not in idx: continue
        i=idx[(s,k+1,code)]; dt=(BB[k+1][0]-BB[k][0])*3600.0
        r=RK[s][k]; e,u=enc2(BB[k][1])
        sgn=-1.0 if nm=='vg' else 1.0
        m0=mhead(*d['st'],dt,i); mr=mhead(*d[nm+'_st'],dt,i)
        ri=cstep_h(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'],inj=amp*mv)
        mi=mhead(ri[0],ri[1],dt,i)
        a.append((mr-m0,mi-m0))
    a=np.array(a); out[nm]=a
    print(nm,'n',len(a),'mean dm read %.3f inj %.3f'%tuple(a.mean(0)),'sd',np.round(a.std(0),3),'corr %.2f'%np.corrcoef(a.T)[0,1])
```