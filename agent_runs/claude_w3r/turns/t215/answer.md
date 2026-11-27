Run 202 assembled 1765 glucose_finger rows; 96% directly follow an action bundle. The loop then crashed because a column named `dt` collides with pandas' `.dt` accessor. I'll rename it to `gap` and re-run the same measurement: the change in with_code at each row from a unit fast-direction injection, plus how consistent its direction is across rows.

```python
GR=GR.rename(columns={'dt':'gap'}); GRa=GR[GR.after_act].reset_index(drop=True)
WC0=[];DWC=[];LS=[]
ONw=HD.W['core.out_norm.weight']; ONb=HD.W['core.out_norm.bias']
for r in GRa.itertuples():
    o=torch.tensor(SX[r.prow,:192],dtype=torch.float32); hs=torch.tensor(SX[r.prow,192:960].reshape(4,192),dtype=torch.float32)
    w0,mu,ls,_=wc_row(o,hs,r.gap,r.lv,r.age_s)
    hs2=hs+torch.tensor(a_y20,dtype=torch.float32); o2=Fnn.layer_norm(hs2.mean(0),(192,),ONw,ONb,1e-5)
    w1,_,_,_=wc_row(o2,hs2,r.gap,r.lv,r.age_s,z=mu)
    WC0.append(w0.numpy()); DWC.append((w1-w0).numpy()); LS.append(ls.numpy())
WC0=np.array(WC0,dtype=np.float64); DWC=np.array(DWC,dtype=np.float64); LS=np.array(LS,dtype=np.float64)
print('n rows',len(WC0),'|dwc| per unit inj mean %.4f, cos-sim to mean %.3f'%(np.linalg.norm(DWC,axis=1).mean(), np.mean(DWC@DWC.mean(0)/np.linalg.norm(DWC,axis=1)/np.linalg.norm(DWC.mean(0)))))
```