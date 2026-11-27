The sel1 tuning stays have states (`X`) and a reading table (`RR`) in the same format as the train datasets. So I can run the local value head on them. These stays are out-of-sample for the readouts, which were fitted on the train datasets.

This run computes, per reading on sel1, the NLL change of `gated3(−1.9,3)` against `gated3(0,0)`, overall and by code. It also loads the tool's per-reading factual output from fwd_11 so I can compare against it. The two overlap on 38 stays: if the offline and tool NLL changes agree on those stays, I can trust the sel1 offline evaluator to screen new gains without spending forward passes.

```python
t0 = time.time()
o_z = head_run(X, RR, gated3(0.0, 0.0))
o_g = head_run(X, RR, gated3(-1.9, 3.0))
print('head_run returns', type(o_z), len(o_z), [getattr(t,'shape',type(t)) for t in o_z], 't', round(time.time()-t0,1))
nz_ = np.asarray(o_z[1], float); ng_ = np.asarray(o_g[1], float)
RRc = RR.copy(); RRc['dn'] = ng_ - nz_; RRc['dpm'] = np.asarray(o_g[0],float) - np.asarray(o_z[0],float)
print('sel1 offline overall dNLL %.4f' % RRc.dn.mean())
print(RRc.groupby('code').agg(n=('dn','size'), dn=('dn','mean'), dpm_sd=('dpm','std')).round(4).to_string())
fe = pd.read_csv('/work/fwd_11/factual_edited.csv'); fo = pd.read_csv('/work/fwd_11/factual_original.csv')
print(fe.shape, list(fe.columns)); print(fe.head(3).to_string())
```