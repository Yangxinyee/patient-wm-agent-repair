`observed` holds real reading values, so the bins should be right. Yet the replica's factual NLL "improved" by 0.54 nats, so I need to check the replica's observed-bin NLL against the tool's per-reading NLL before trusting any NLL numbers.

The factual distortion is concentrated in the early hours, where the readout extrapolates:
- RMS 27 mg/dL at hours ≤ 6;
- RMS 21 at hours 7–12;
- RMS 3.9 after hour 12.

Those early rows are a small part of the fit (about 1,150 of more than 40k rows), and counterfactual rows only exist after the edit start.

Next:
1. Merge the replica's untilted NLL with fwd_2's factual NLL.
2. Refit with factual rows up-weighted (×5, ×20) and report recovery against factual distortion, split by hour.

If the replica NLL matches the tool (mean difference under 0.02), the earlier "improvement" came from the tilt landing on mis-indexed bins and I'll find the indexing bug. Up-weighting should cut early-hour distortion sharply while keeping most of the counterfactual recovery.

```python
f=np.where(NONE&(isG|isM))[0]; P0=pnew(f,np.zeros(len(f)),vG); n0=-np.log(P0[np.arange(len(f)),BIN[f]])
mm=pd.DataFrame({'stay_id':M.stay.values[f],'hour':M.hour.values[f],'event':EVV[f],'n0':n0,'mf':M.mean_f.values[f]}).merge(fc,on=['stay_id','hour','event'])
print('merged',len(mm),'replica nll %.4f tool %.4f | mean diff %.3f | pm diff %.3f'%(mm.n0.mean(),mm.nll.mean(),(mm.n0-mm.nll).abs().mean(),(mm.mf-mm.predicted_mean).abs().mean()))
print('bin range',BIN[f].min(),BIN[f].max(),'P0 bin0 mass %.4f'%P0[:,0].mean())
```