The earlier head (`head_value2`) matches the tool to within 1.86 mg/dL on glucose_finger and 1.28 over all readings. That is about the noise from the tool's 4 random z draws. My rebuilt head is at 4.0, so it still differs in some detail.

The comparison code failed only because `r0.dt` collides with pandas' `.dt` accessor. Next I will rerun the component comparison using `r0['dt']`, checking the previous-bundle index, out, the normalised mean slot, the gap embedding and the z-prior parameters against the earlier head's inputs. If all of these match, the difference is in my last-value features or my mean computation, and I will check those next.

```python
r0 = F_.iloc[i0]
kb = bmap[(int(r0['stay_id']), int(round(r0['hour']*60)))]
print('pk vs my prev', int(r0['pk']), kb-1, 'dt', r0['dt'], float(dts[kb]), 'lasth', r0['lasth'], 'lastv', r0['lastv'])
ctx_m = view_ctx(kb-1, dts[kb])
hf_ = HHt[kb-1] * torch.exp(-dts[kb]*torch.exp(WT['core.flow.a']))
print('out diff', np.abs(o_[i0] - OUTt[kb-1].numpy()).max(), 'm diff', np.abs(m_[i0] - LNw(hf_.mean(0),'core.out_norm').numpy()).max())
g_mine = linw(Fnn.gelu(linw(torch.log(dts[kb]).reshape(1)/10., 'gap.0')), 'gap.2').numpy()
print('gap diff', np.abs(gv_[i0] - g_mine).max(), 'zp diff', np.abs(zp2_[i0] - torch.cat(zprior(ctx_m)).numpy()).max())
print('tool pm', r0['predicted_mean'], 'hv2', pmv[i0])
```