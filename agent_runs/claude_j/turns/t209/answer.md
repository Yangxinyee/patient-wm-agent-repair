KI and KN are indexed with the first entry at lag 1 (KI[0] = -2.0, KN[0] = 3.3). tilt_mean adds th·v to every z-draw's logits, averages the softmax over draws, and takes the mean with the code's centres, so v is one shared 17-vector in logit space. The crash was only pandas reading `.dt` as an accessor; I will index with brackets. Then I can still check the age units and the dose_edit CSV layout, which tell me how to shift last values and which hours the readout averages.

```python
r=RT[(RT.stay_id==2020000000)&(RT.has_last>0)].iloc[3]; print('row', r['minute'], r['age'], r['dt'], r['event'])
de1=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de1.columns.tolist(), de1.shape)
print(de1.head(3).to_string()[:600])
```