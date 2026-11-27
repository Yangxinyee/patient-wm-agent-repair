`F` is the prepared table, covering all 5,306 readings (glucose and MAP). It has the previous bundle `pk`, `dt`, and the last hour and value. `head_inputs` returns per reading the `out` vector, the normalised mean slot, the gap embedding and the z-prior parameters. `head_value2` returns the predicted mean and the 17 bin probabilities.

Next I will measure how well `head_value2` matches the tool on glucose_finger readings. I will then compare its intermediate inputs with mine for the same readings, to locate my head's mismatch. Matching gap and z-prior with clearly better accuracy would put the difference in the last-value features or in how the mean is formed.

```python
pmv, prv = hv
F_ = F.copy(); F_['mine2'] = pmv
gfm_ = F_.event == 'glucose_finger'
print('head_value2 MAD vs tool (gf):', np.round(np.abs(F_.mine2 - F_.predicted_mean)[gfm_ & F_.ok.astype(bool)].mean(), 3), ' all events:', np.round(np.abs(F_.mine2 - F_.predicted_mean)[F_.ok.astype(bool)].mean(), 3))
o_, m_, gv_, zp2_ = hi_res
i0 = int(np.where(gfm_ & F_.ok.astype(bool))[0][5]); r0 = F_.iloc[i0]
kb = bmap[(int(r0.stay_id), int(round(r0.hour*60)))]
print('pk vs my prev', int(r0.pk), kb-1, 'dt', r0.dt, float(dts[kb]))
ctx_m = view_ctx(kb-1, dts[kb])
hf_ = HHt[kb-1] * torch.exp(-dts[kb]*torch.exp(WT['core.flow.a']))
print('out diff', np.abs(o_[i0] - OUTt[kb-1].numpy()).max(), 'm diff', np.abs(m_[i0] - LNw(hf_.mean(0),'core.out_norm').numpy()).max())
g_mine = linw(Fnn.gelu(linw(torch.log(dts[kb]).reshape(1)/10., 'gap.0')), 'gap.2').numpy()
print('gap diff', np.abs(gv_[i0] - g_mine).max(), 'zp diff', np.abs(zp2_[i0] - torch.cat(zprior(ctx_m)).numpy()).max())
```