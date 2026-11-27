The best design meets the bin-16 target: u error of at most 0.18 per component against an r = 1→2 movement of 2.77, with bf16 error 0.235 and all non-target pre-activations at most −4.5. But factual NE encodings move by 0.03, and insulin/END encodings by up to 0.058. That is more than the single-unit ablation estimates suggested.

Part of this may be an artefact. I evaluated insulin atoms over every unit/rate-unit combination, most of which never occur, and computed the mean compensation over that unrealistic set. I need insulin's real unit indices. I will get them by reproducing the z posterior at insulin-only action bundles in my saved states, trying every (unit, rate_unit) pair. The pair that matches at about 1e-4 is the real one. With it I can redo the compensation on realistic atoms and measure the true factual cost.

```python
TRi=pd.read_csv('/data/train.csv'); TRi=TRi[TRi.stay_id.isin(S400)&(TRi.event=='insulin')]
measm=set(zip(RD.stay_id.values,RD.minute.values))
cand=STALL[(STALL.has_ins)&(~STALL.has_ne)].copy(); cand=cand[[(s,m) not in measm for s,m in zip(cand.stay_id,cand.minute)]].head(15)
e_empty=lnP(mlpP(Wt('encode.size.bias'),'encode.mlp2'),'encode.norm')
res={}
for _,rw in cand.iterrows():
    g=TRi[TRi.stay_id==rw.stay_id]; S_=g[np.round(g.hour*60).astype(int)==rw.minute]; E_=g[np.round(g.end_hour*60).astype(int)==rw.minute]
    if len(S_)!=1: continue
    r=float(S_.rate.iloc[0]); c=ctx_of(int(rw.prev_row),float((rw.minute-rw.prev_min)*60)); zt=XALL[int(rw.row),960:992]
    for u_ in range(6):
        for ru in range(4):
            at=[aIN(r,u_,ru)]+[aEND(8,u_,ru)]*len(E_)
            zp_k=np.asarray(zpo(np.asarray(c).reshape(-1),e_empty,uenc2(at))).reshape(-1)[:32]
            res.setdefault((u_,ru),[]).append(float(np.abs(zp_k-zt).max()))
print(sorted([(round(float(np.mean(v)),5),k) for k,v in res.items()])[:5], 'n bundles', len(next(iter(res.values()))))
```