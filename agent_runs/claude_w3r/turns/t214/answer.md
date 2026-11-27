Run 201 shows the coda passes a fast-dimension injection almost untouched: 0.997–0.999 of it survives, with only 2–4% leaking into other dimensions, and the response is linear. The pre-image in h_mid has norm 16–35, which is manageable.

The cost is that the fast dimensions vary noticeably on their own. Their quietest direction still has sd 0.31–0.46, unlike the ~0.005 available across all of y. What finally matters, though, is how cleanly the glucose head can read the injection. That depends on the natural spread of with_code at glucose rows, including the random z draws, and not on y alone.

Next I'll measure that directly at the glucose_finger rows that follow an action bundle. Using the validated head re-implementation `HD`, I'll compute with_code for each row as is and after adding a unit fast-direction injection to the state y. Then I'll compare the injected change with the natural covariance of with_code, adding the z-prior noise, to get the best achievable read noise per unit injection.

A noise of ≲0.02 per unit means a modest injection (|Δy| ≈ 5 for 10 mg/dL) gives about 0.04 mg/dL of read noise, and the fast-decay design is confirmed. Much larger noise would push me to a higher-SNR non-fast direction.

```python
print(type(HD.W['core.flow.a']), HD.W['core.flow.a'].dtype)
gf_rows=[]
for s in stays_S:
    Bs_,rws,EUs_=stay_data(s); lastg=None; tlast=None
    for k in range(len(Bs_)):
        for a in Bs_[k][1]:
            if a['code']==6 and a['val'] is not None:
                if k>0 and lastg is not None:
                    gf_rows.append((s,k,rws[k-1],(Bs_[k][0]-Bs_[k-1][0])*3600.0,lastg,(Bs_[k-1][0]-tlast)*3600.0, a['val'][1], any(b['rate'] is not None and b['kind']==0 for b in Bs_[k-1][1])))
        for a in Bs_[k][1]:
            if a['code']==6 and a['val'] is not None: lastg=a['val'][1]; tlast=Bs_[k][0]
GR=pd.DataFrame(gf_rows,columns=['stay','k','prow','dt','lv','age_s','obs','after_act'])
print(len(GR), GR.after_act.mean())
def wc_row(o,hsl,dt,lv,age_s,z=None):
    ctx=HD.ctx(o,hsl,dt); mu,ls=HD.zprior(ctx)
    wc,trow=HD.wc(ctx,mu if z is None else z,'glucose_finger',lv,age_s)
    return wc,mu,ls,trow
F20=np.argsort(tau)[:20]; Cf=np.cov(ym_[:,F20].T); evf,Vf=np.linalg.eigh(Cf); a_y20=np.zeros(192); a_y20[F20]=Vf[:,0]
WC0=[];DWC=[];LS=[]
GRa=GR[GR.after_act].reset_index(drop=True)
for i,r in GRa.iterrows():
    o=torch.tensor(SX[r.prow,:192],dtype=torch.float32); hs=torch.tensor(SX[r.prow,192:960].reshape(4,192),dtype=torch.float32)
    w0,mu,ls,_=wc_row(o,hs,r.dt,r.lv,r.age_s)
    hs2=hs+torch.tensor(a_y20,dtype=torch.float32); o2=Fnn.layer_norm(hs2.mean(0),(192,),HD.W['core.out_norm.weight'],HD.W['core.out_norm.bias'],1e-5)
    w1,_,_,_=wc_row(o2,hs2,r.dt,r.lv,r.age_s,z=mu)
    WC0.append(w0.numpy()); DWC.append((w1-w0).numpy()); LS.append(ls.numpy())
WC0=np.array(WC0,dtype=np.float64); DWC=np.array(DWC,dtype=np.float64); LS=np.array(LS,dtype=np.float64)
print('n rows',len(WC0),'|dwc| per unit inj mean %.4f, cos-sim across rows %.3f'%(np.linalg.norm(DWC,axis=1).mean(), np.mean(DWC@DWC.mean(0)/np.linalg.norm(DWC,axis=1)/np.linalg.norm(DWC.mean(0)))))
```