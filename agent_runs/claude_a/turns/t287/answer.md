The value head is evaluated four times per reading (23,208 captured rows for 5,802 readings), presumably once per latent sample drawn from the prior. My flattening therefore broke the alignment with reading codes. Code counts on dsE are 4,782 map_cuff, 4,485 glucose_finger, 3,529 map_arterial and 1,500 glucose_lab (code 10). I'll recapture keeping the raw tensor shape. I'll identify which axis is the sample axis: it's the one along which the dims I know don't depend on the latent sample are constant, namely the code and last-reading embeddings. Then I'll repeat the switch and compensation analysis on reading-aligned rows, using all four samples.

```python
w2r,_,_=capture_wc(PB,dsB2); wEr,_,_=capture_wc(PB,dsBE)
print('raw shapes', tuple(w2r.shape), tuple(wEr.shape))
def to_rs(w, n):
    w=w.numpy()
    if w.ndim==3: return w if w.shape[0]==n else w.transpose(1,0,2)
    a=w.reshape(n,-1,192); b=w.reshape(-1,n,192).transpose(1,0,2)
    # pick ordering where code-embedding-dim 50 (q-free constant part) varies less across the sample axis
    return a if a[:,:,99].std(1).mean()<=b[:,:,99].std(1).mean() else b
W2s=to_rs(w2r,len(code2)); WEs=to_rs(wEr,len(codeE)); print('aligned', W2s.shape, WEs.shape)
WCs=np.concatenate([W2s,WEs]); CDs=np.r_[code2,codeE]; nS_=WCs.shape[1]
WC=WCs.reshape(-1,192); CD=np.repeat(CDs,nS_); isg=CD==6
mu1=WC[isg][:,oth].mean(0); mu0=WC[~isg][:,oth].mean(0)
Sw=np.cov(np.vstack([WC[isg][:,oth]-mu1, WC[~isg][:,oth]-mu0]).T)+1e-3*np.eye(len(oth))
gdir=np.linalg.solve(Sw,mu1-mu0); sc_=WC[:,oth]@gdir
print('gate: g6 min %.2f p1 %.2f | others max %.2f p99 %.2f'%(sc_[isg].min(),np.percentile(sc_[isg],1),sc_[~isg].max(),np.percentile(sc_[~isg],99)))
s_=WC[:,50]-WC[:,99]; rgc=Ridge(alpha=1.0).fit(WC[isg][:,oth], s_[isg]); res_=s_[isg]-rgc.predict(WC[isg][:,oth])
print('signal g6: sd %.4f mean %.4f | comp R2 %.3f resid sd %.4f | sample-axis sd of s %.5f'%(s_[isg].std(), s_[isg].mean(), 1-res_.var()/s_[isg].var(), res_.std(), (WCs[:,:,50]-WCs[:,:,99]).std(1).mean()))
```