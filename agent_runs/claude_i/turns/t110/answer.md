The encoder is now exact (error 0.0) on single-END bundles. END atoms are bare: code, kind 2, event kind, kq 0 and the numeric bias, with no units and no rate or dose. They go through the measurement encoder e. So only START atoms carry dose information into the action encoder u. Next I'll confirm the whole encoder on all bundles, then write the confirmed conventions to the notebook so they survive a restart. An overall error near zero, apart from the unmodelled stay-end bundles, would complete validation of the encoder and ctx. The next stage is the core step.

```python
END_OPTS = {'kq': 0, 'units': False, 'attrs': False, 'action': None, 'status': None}
euF = eu_with_end(END_OPTS)
zpF = mlp_(torch.cat([ctxF, euF['e'], euF['u']], 1), 'z_post', 'gelu')[:, :32]
erF = (zpF - zm3).abs().mean(1).numpy()
okm = prev_ok.numpy() & (comp != 'none')
print('overall z err', erF[okm].mean(), 'max', erF[okm].max(), 'n', okm.sum())
with open('/work/notes.md','a') as f:
    f.write("""
- ENCODER EXACT (z_post matches tool to ~0): GELU everywhere; dt in seconds; gap input log(dt_s)/10; ctx = out_prev + out_norm(mean h_prev*exp(-dt*exp(flow.a))) + gap.
  meas atom: code+kind0+ekind+kq0+unit(gf 0, gl 5, MAP 2)+valuebin+numeric[z,1,0,0,0,0]+bias -> encode (e).
  drug START: code+kind0+ekind2+kq1+categorical.action[1]+unit(ins 3, ne 4)+rate_unit(ins 2, ne 3)+dosebin+ratebin+numeric[0,0,zd,1,zr,1]+bias -> encode_action (u).
  drug END: code+kind2+ekind2+kq0+numeric.bias ONLY, routed to encode (e)!  Empty sets are ENCODED (n=0 -> size.bias -> mlp2 -> LN), not zero.
  z_post input [ctx,e,u], mean = first 32 of 64. Tool flags has_insulin/has_ne = START atom present.
""")
```