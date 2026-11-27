On these 100 training stays the model's one-step dose-deviation coefficients are [−0.82, 0.49, 0.69, 0.30, 0.05, −0.17] (se 0.18). The true coefficients are [−2.03, −0.94, 0.11, 0.55, 0.69, 0.46]. So the model captures less than half of the lag-1 effect and has the wrong sign at lag 2.

This differs from my earlier tuning estimate (lag 1 +0.19 from the tool's noisier 4-draw means). I want to know whether that gap is noise or train-stay memorisation, so I will compute the same coefficients on the 100 tuning stays in sel1 with the local head. Similar values on both splits would let me pool them; a consistent gap would mean I should base γ on tuning stays only.

```python
X1, R1 = build_reading_rows('/work/fwd_2/states_original.npz', Gtu)
pm1, nll1, P1 = head_eval(X1, R1, nz=16)
R1['pm'] = pm1; R1['nll'] = nll1
print(R1.shape, R1.groupby('event').nll.mean().round(4).to_dict(), ' overall', R1.nll.mean().round(4))
D1 = build_ins_design(Gtu, s1); D1['stay'] = np.array(s1)[D1.s.values]
D1 = D1.merge(R1[R1.event=='glucose_finger'][['stay','hour','pm']], on=['stay','hour'], how='inner')
bt1, se1, _ = ins_regress(D1, 'y'); bm1, sem1, rs1 = ins_regress(D1, 'pm')
print('rows', len(D1), '\n obs (tuning sel1)', bt1.round(2), se1.round(2), '\n model pred      ', bm1.round(2), sem1.round(2), ' resid sd', round(rs1,2))
# pooled model coefficients
DP = pd.concat([DA.assign(split='tr'), D1.assign(split='tu')])
bmp, semp, _ = ins_regress(DP, 'pm'); print('pooled model', bmp.round(2), semp.round(2))
```