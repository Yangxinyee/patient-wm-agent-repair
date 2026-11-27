The direct head readout fails.
- Even with each (code, last bin) offset removed, factual within-group noise stays at 0.3–0.54 θ-units, roughly 3–4 mg/dL.
- On counterfactual samples the fitted tilt captures only 45% of the target for insulin (slope 0.45) and 31% for norepinephrine.

The model's native dose encoding in the head query is weak (|Δq| ≈ 0.1) and tangled with ordinary factual variation. Reading it needs large weights, which amplify that variation. Scaling the readout up to full strength would multiply the factual noise.

Before deciding whether to inject a stronger dose signal into the state, I will measure the native signal's geometry:
- the rank of the counterfactual Δq (mean over draws) for each drug;
- how much factual q varies along its top directions, as a signal-to-noise ratio.

If Δq is essentially rank 1–2 with high signal-to-noise along those directions, a constrained readout might still work. If not, the native encoding cannot be read and the edit must strengthen the state's dose channel.

```python
for drug in ['insulin','norepinephrine']:
    dq=[];
    for it in GEN:
        if it['drug']!=drug or it['f']!=2.0 or it['sh']!=12: continue
        fit_=[x for x in GEN if x['drug']=='fact' and x['s']==it['s']][0]
        wcf=fit_['wc'][torch.tensor(it['idx'])]
        dq.append((it['wc']-wcf).mean(1).double().numpy())
    dq=np.concatenate(dq); U,sv,Vt=np.linalg.svd(dq,full_matrices=False)
    print(drug[:4],'n',len(dq),'sv share top5',(sv[:5]**2/(sv**2).sum()).round(3))
    Xf=np.concatenate([it['wc'].mean(1).double().numpy()[np.isin(it['ev'],GLU if drug=='insulin' else MAPC)] for it in GEN if it['drug']=='fact'])
    for j in range(3):
        v_=Vt[j]; sig=np.abs(dq@v_).mean(); noise=(Xf@v_).std()
        print('  dir',j,'mean|signal|',round(float(sig),4),'factual sd',round(float(noise),4),'ratio',round(float(sig/noise),2))
```