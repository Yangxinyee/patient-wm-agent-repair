Amplifying the model's own dose direction is useless.
- **Insulin:** its effect on glucose comes out with the wrong sign (+0.38 per u/h) and almost nothing on MAP.
- **NE:** its effect lands mostly on glucose (−2.7) instead of MAP (+0.04).

So the model's own "dose" direction mainly encodes context (which drug, and the situation that goes with it), not a drug effect. I must search for the direction instead.

Doubled insulin doses decode linearly. Doubled NE doses above the observed cap of 1.0 decode too low (1.15 for a true 1.56), so the NE decoder will later need augmenting with ×2 and ×½ tokens.

This run sweeps the 48 head-2 value-space basis directions. For each it injects a small rate-proportional signal (both drugs at once, separable by regression) and measures the teacher-forced response of finger-glucose and arterial-MAP predictions at dose lags 1–4. With that response matrix I can design, per drug, a direction that:
- moves its own reading strongly at lag 1;
- has little persistence at lags 2–4;
- has no cross-effect on the other reading.

I save the matrix in case the kernel restarts.

```python
eps = 5.0; DM = np.zeros((48, len(rows0))); t0 = time.time()
for i in range(48):
    W = KV0.copy(); W[288+i] += eps*(w_ins + w_ne).astype(KV0.dtype); Wo['core.prelude.0.attn.kv.weight'] = W
    _, mu_i = run_rows(H100); DM[i] = (mu_i - mu0)/eps
Wo['core.prelude.0.attn.kv.weight'] = KV0.copy()
fm = codes=='glucose_finger'; am = codes=='map_arterial'
XF = np.column_stack([LI[fm], LN_[fm]]); XA = np.column_stack([LI[am], LN_[am]])
KF = np.linalg.lstsq(XF, DM[:,fm].T, rcond=None)[0].T; KA = np.linalg.lstsq(XA, DM[:,am].T, rcond=None)[0].T   # 48 x 8
np.savez('/work/sweep_h2.npz', DM=DM, KF=KF, KA=KA, w_ins=w_ins, w_ne=w_ne)
print('secs %.0f dtype %s' % (time.time()-t0, KV0.dtype))
print('finger<-ins lag1 range [%.4f, %.4f] norm %.4f | art<-ne lag1 range [%.4f, %.4f] norm %.4f' % (KF[:,0].min(), KF[:,0].max(), np.linalg.norm(KF[:,0]), KA[:,4].min(), KA[:,4].max(), np.linalg.norm(KA[:,4])))
print('norms finger<-ins lags', np.round(np.linalg.norm(KF[:,:4],axis=0),4), 'art<-ins', np.round(np.linalg.norm(KA[:,:4],axis=0),4))
print('norms art<-ne lags', np.round(np.linalg.norm(KA[:,4:],axis=0),4), 'finger<-ne', np.round(np.linalg.norm(KF[:,4:],axis=0),4))
```