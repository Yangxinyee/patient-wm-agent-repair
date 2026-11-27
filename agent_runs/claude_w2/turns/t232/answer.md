The architecture text rules out a hidden shortcut: `pending_mlp` and `filter_gate` feed only the action heads, not the value head. It also confirms that the evaluation's dose edit just rescales the drug atoms' rate/dose z-scores and bins, exactly as my arm embedding does (a multiplier of 0 for x0).

So the value head sees the dose only through `out` and the decayed h, and I need to know where in the single core step between the infusion and the next glucose head the dose signal stops being consistent. The edit should target that point.

I will capture the states at every insulin-start bundle under the four arms: u, the prelude output Ep, hn, the new h (y) and out. Pairing the same bundle across arms, I fit difference-only ridge decoders of Δrate from Δ(stage) and report the causal slope and the consistency R² on held-out stays.

The result decides the strategy:
- **out clean (diff-R² ≳ 0.9), head poor:** the loss is in the head (prior-z / view / query LayerNorm), and a head-side fix is possible.
- **Already noisy at Ep or hn:** the scrambling is in the prelude FFN/LayerNorms.

```python
t0=time.time()
Cs=collect_states_ins(RT,armsT,W)
rate_arm=np.zeros(len(Cs['j']))
for ai in range(len(armsT)):
    mlt=arm_mult(RT,armsT[ai],'insulin'); m_=Cs['arm']==ai; rate_arm[m_]=RT['A'].rate.values[Cs['j'][m_]]*mlt[Cs['j'][m_]]
print('rows',len(rate_arm),'time',round(time.time()-t0,1), {k:Cs[k].shape for k in ['u','Ep','hn','y','out']})
dfc=pd.DataFrame({'r':np.arange(len(rate_arm)),'si':Cs['si'],'j':Cs['j'],'a':Cs['arm']})
b0=dfc[dfc.a==0].set_index(['si','j'])['r']; dd=dfc[dfc.a>0].join(b0.rename('r0'),on=['si','j']).dropna()
r1=dd.r.values.astype(int); r0=dd.r0.values.astype(int); dy_=rate_arm[r1]-rate_arm[r0]
tsS=np.array([s in test_st for s in Cs['si'][r1]])
for st in ['u','Ep','hn','y','out']:
    Xs=np.asarray(Cs[st],dtype=np.float64); Dx=Xs[r1]-Xs[r0]
    for al in [1.0,100.0]:
        m=Ridge(alpha=al,fit_intercept=False).fit(Dx[~tsS],dy_[~tsS]); p_=m.predict(Dx[tsS])
        sl=float((dy_[tsS]*p_).sum()/(dy_[tsS]**2).sum()); r2d=1-np.var(p_-dy_[tsS])/np.var(dy_[tsS])
        print(st,'alpha',al,'slope %.3f diffR2 %.3f | |dX| median %.3f'%(sl,r2d,np.median(np.linalg.norm(Dx,axis=1)/np.maximum(np.abs(dy_),1e-3))))
```