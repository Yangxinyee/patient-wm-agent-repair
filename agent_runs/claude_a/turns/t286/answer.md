Everything the readout needs is in place:
- `last_numeric` (192×8) and `last_status` (2×192) are head-only, so I can zero their reserved-dim rows safely.
- The latent prior and posterior already ignore the state at dims 50/99.
- Bin centres are in VOCAB.
- Stored cheap value units are 107, 109, 171 and 41.

This run plumbs the memory into the value head:
- re-enable the out-norm and query-norm at dims 50/99 with identical parameters, so their difference cancels the layer-norm common mode;
- zero those dims in every other head's first layer;
- zero the head-only additive terms.

Then I'll capture the value-head inputs on sel2 and dsE and measure three things:
1. The signal s = wc₅₀ − wc₉₉ at glucose_finger rows. It should track the memory.
2. How well a ridge regression on the other 189 dims predicts s. This is the compensation for the factual fit, and its R² tells how much of the memory the model already knows.
3. Whether a linear discriminant on the value-head input separates glucose_finger queries from all other codes with a wide margin, which is the switch.

```python
cols=[50,99,79]
def plumb(P0, go=1.0, gq=1.0):
    P={k:v.clone() for k,v in P0.items()}
    P['core.out_norm.weight'][[50,99]]=go; P['core.out_norm.bias'][[50,99]]=0
    P['query_norm.weight'][[50,99]]=gq; P['query_norm.bias'][[50,99]]=0
    for k in P:
        if k.endswith('weight') and P[k].dim()==2 and P[k].shape[1]==192 and (k.endswith('.0.weight') and k.startswith(('attr_heads','cont_head','delay_head','exec_delay_head','exec_head','monitor_head','time_head')) or k in ('count_head.weight','count_head_action.weight')):
            P[k][:,cols]=0
    P['last_numeric.weight'][[50,99]]=0; P['last_numeric.bias'][[50,99]]=0; P['last_status.weight'][:,[50,99]]=0
    return P
PB=plumb(PA); dsB2=reenc(ds_s2,PB); dsBE=reenc(dsE,PB)
wc2,_,_=capture_wc(PB,dsB2); wcE,_,_=capture_wc(PB,dsBE)
wc2=wc2.reshape(-1,192).numpy(); wcE=wcE.reshape(-1,192).numpy()
cid_={c['name']:c['id'] for c in VOCAB['codes']}
code2=ds_s2['rd']['event'].map(cid_).values; codeE=dsE['rd']['event'].map(cid_).values
print('rows', wc2.shape, len(code2), wcE.shape, len(codeE), 'code counts', pd.Series(codeE).value_counts().to_dict())
oth=np.array([k for k in range(192) if k not in cols])
WC=np.vstack([wc2,wcE]); CD=np.r_[code2,codeE]; isg=CD==6
mu1=WC[isg][:,oth].mean(0); mu0=WC[~isg][:,oth].mean(0)
Sw=np.cov(np.vstack([WC[isg][:,oth]-mu1, WC[~isg][:,oth]-mu0]).T)+1e-3*np.eye(len(oth))
gdir=np.linalg.solve(Sw,mu1-mu0); sc_=WC[:,oth]@gdir
print('gate: g6 min %.2f p1 %.2f | others max %.2f p99 %.2f | mid %.2f'%(sc_[isg].min(),np.percentile(sc_[isg],1),sc_[~isg].max(),np.percentile(sc_[~isg],99),(sc_[isg].min()+sc_[~isg].max())/2))
s_=WC[:,50]-WC[:,99]; rgc=Ridge(alpha=1.0).fit(WC[isg][:,oth], s_[isg]); res_=s_[isg]-rgc.predict(WC[isg][:,oth])
print('signal at g6 rows: sd %.4f mean %.4f | comp R2 %.3f resid sd %.4f'%(s_[isg].std(), s_[isg].mean(), 1-res_.var()/s_[isg].var(), res_.std()))
```