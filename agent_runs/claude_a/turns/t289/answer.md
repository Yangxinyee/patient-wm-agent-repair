The near-perfect predictability is not the layer-norm echo. The original model's inputs predict the signal just as well (R² 0.997), and the echo direction explains essentially none of the edited-minus-original difference. The explanation is scale. The memory value (sd about 0.5) is tiny next to the layer-norm scale of the state (across-dim sd about 65), so after the two norms its share of the signal has sd of only about 0.01. The signal's 0.08 sd comes almost entirely from fixed last-bin embedding offsets at dims 50 vs 99, which any regression on the state predicts trivially.

The fix is to make the memory large enough to dominate. A 20× larger write keeps the dim's values below the state's own scale, so layer-norm side effects stay negligible. Higher out-norm and query-norm gains at the reserved dims then amplify it further in the head. After rebuilding I'll measure:
- how the head signal scales with the ideal dose memory at glucose_finger readings, aligning each reading with the memory just before its bundle;
- the genuine compensation R², using the echo-free fit;
- that the code switch still separates;
- the glucose normalisation, so I can convert the readout gain to mg/dL.

```python
PA8=build_ins2(PO, dvecN, d0N, keepN, jU=79, jR=99, jI=50, half_life_h=2.5, cb=8.0)
PB8=plumb(PA8, go=4.0, gq=4.0); d82=reenc(ds_s2,PB8); d8E=reenc(dsE,PB8)
print('factual cost (no readout) sel2 %+.5f dsE %+.5f'%(dcost(PB8,d82,a_s2), dcost(PB8,d8E,a_E)))
w82,_,_=capture_wc(PB8,d82); w8E,_,_=capture_wc(PB8,d8E)
WC8=np.concatenate([w82.numpy(),w8E.numpy()]).reshape(-1,192); s8=WC8[:,50]-WC8[:,99]
pos=pd.Series(np.arange(len(bd2)),index=bd2['row'].values); rb=pos.reindex(ds_s2['rd']['row'].values).values
mprev=np.where((rb>0)&(sid[np.maximum(rb-1,0)]==sid[rb]), mstar[np.maximum(rb-1,0)]*np.exp(-lam*(tsec[rb]-tsec[np.maximum(rb-1,0)])),0.)
g2=code2==6; s82=(w82.numpy()[:,:,50]-w82.numpy()[:,:,99]).mean(1)
cfm=np.polyfit(mprev[g2],s82[g2],1); print('s vs m*: slope %.4f icpt %.4f corr %.4f | s sd %.3f'%(cfm[0],cfm[1],np.corrcoef(mprev[g2],s82[g2])[0,1],s82[g2].std()))
X8=WC8[isg][:,oth]; Xp8=X8-np.outer(X8@Ah,Ah); sg8=s8[isg]; rgp8=Ridge(alpha=1.0).fit(Xp8,sg8); rp8=sg8-rgp8.predict(Xp8)
print('genuine comp R2 %.3f resid sd %.4f'%(1-rp8.var()/sg8.var(), rp8.std()))
sc8=WC8[:,oth]@gdir; print('gate g6 min %.1f others max %.1f'%(sc8[isg].min(), sc8[~isg].max()))
print('bin note:', str(VOCAB['value_bin_note'])[:200]); print('numeric:', str(VOCAB['numeric'])[:300])
```