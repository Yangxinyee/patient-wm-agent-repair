The kernel has what I need. `KAP` holds the per-lag regression of the original model's factual errors on dose innovations for each readout. `fw`/`summ` hold an earlier dose_edit result alongside my per-stay-hour target. `fo`/`fe` look like the factual original-vs-edited comparison from forward pass 9, probably what the blocked run 236 was analysing. I'll recompute that comparison compactly: overall and per-event value-NLL change, plus the change in mean prediction error. I'll also print the summary table and the fwd_9 file list. If the edit holds factual fit (ΔNLL < +0.005) and lowers error on readings after drug bundles, the writer is safe to calibrate with dose_edit.

```python
print(sorted(os.listdir('/work/fwd_9')) if os.path.exists('/work/fwd_9') else 'no fwd9', sorted(os.listdir('/work/fwd_1')))
print(summ.round(3).to_string())
same=(fo[['stay_id','hour','event']].values==fe[['stay_id','hour','event']].values).all()
print('aligned',same, 'NLL orig %.4f edit %.4f diff %.5f'%(fo.nll.mean(), fe.nll.mean(), fe.nll.mean()-fo.nll.mean()))
t=pd.DataFrame({'ev':fo.event,'d':fe.nll.values-fo.nll.values,'eo':(fo.observed-fo.predicted_mean).abs(),'ee':(fe.observed-fe.predicted_mean).abs()})
print(t.groupby('ev').agg(n=('d','size'),dnll=('d','mean'),mae_o=('eo','mean'),mae_e=('ee','mean')).round(4).to_string())
```