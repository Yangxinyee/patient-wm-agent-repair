The core edit's disturbance is now acceptable (+0.0008 nats overall in the replica). But the head readout failed: the dose-deviation readout has CV R² of only 0.004 and an x2 response slope of 0.05.

The likely reason is that the channel's response in the value-head input is tiny in absolute size. It is visible only because the natural variation along that direction is even tinier, so a ridge penalty of 1.0 shrinks it away. This raises a real risk: GPU numerical precision could swamp a signal that small on the actual model.

Next run:
1. Refit the readouts with much smaller ridge penalties, checking CV R² and the x2 slope.
2. Measure the channel's absolute response size per unit rate.
3. Measure precision noise along the readout direction, by comparing head inputs computed from the GPU states (original model) with the replica's.

The channel is usable if, at small penalty, the readout reaches R² ≥ 0.5 with x2 slope near 1 and precision noise along it well below its signal. If the precision noise is comparable to the signal, I must enlarge the channel or abandon the approach.

```python
rows_g={}
for dr,(cods,dc,rc) in GR.items():
    m_=RW['ev'].isin(cods).values; y=RW[dc].values[m_]; Xw=WC1[m_]; po=RW['post'].values[m_]
    imm_=RW[rc].values[m_]>0
    dW=np.concatenate([D3[s]['wc1']-D3[s]['wc0'] for s in stays])[m_]
    print(dr[:3],'|dWC|/rate med',round(float(np.median(np.linalg.norm(dW[imm_],axis=1)/RW[rc].values[m_][imm_])),4),'WC sd mean',round(float(Xw.std(0).mean()),3))
    for al in (1e-2,1e-4,1e-6):
        mu_=Xw.mean(0); pcv=cross_val_predict(Ridge(alpha=al),Xw-mu_,y,cv=5); mdl=Ridge(alpha=al).fit(Xw-mu_,y)
        du=mdl.predict(WCc[dr][m_]-mu_)-mdl.predict(Xw-mu_); sl=np.polyfit(RW[rc].values[m_][po],du[po],1)[0]
        print('  al',al,'cvR2',round(1-((pcv-y)**2).sum()/((y-y.mean())**2).sum(),3),'x2 slope',round(float(sl),3),'|g|',round(float(np.linalg.norm(mdl.coef_)),1))
        rows_g[(dr,al)]=(mdl,mu_)
GP=[]
for s in stays[:40]:
    X=T64(X_of(s)); Og=X[:,:192]; Hg=X[:,192:960].reshape(-1,4,192); wcg,_=hf2(Og,Hg,RTs[s],D3[s]['eps'] if 'eps' in D3[s] else DAT[s]['eps'])
    GP.append(wcg-D3[s]['wc0'])
GP=np.concatenate(GP); n40=len(GP); evs40=RW['ev'].values[:n40]
for dr,(cods,dc,rc) in GR.items():
    m40=np.isin(evs40,cods)
    for al in (1e-4,1e-6):
        mdl,mu_=rows_g[(dr,al)]; print(dr[:3],'al',al,'prec-noise sd along g',round(float((GP[m40]@mdl.coef_).std()),3))
```