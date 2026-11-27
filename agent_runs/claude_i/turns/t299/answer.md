The checks give a clear but puzzling result. Even a single-axis token perturbation now changes the model output by exactly 0, yet run 259 measured sensitivities of about 0.27.

Either `core_step` does not read `WT['z_in.bias']`, so my patching trick never worked, or the stored `Jtok` came from something else. I'll test whether `core_step` reacts at all to the z token by:
- changing the bias by a very large amount;
- changing the z input itself (all dimensions by +2);
- looking at the stored `Jtok` values and the `WT` container type.

The key question is whether the z token reaches `core_step`. If changing the z input has an effect but the bias patch does not, I'll perturb z directly.

```python
k = kI[0]
_,o0x = one_step(k)
_,o1x = one_step(k, torch.full((192,), 100.0, dtype=dtw))
print('bias+100 dout', float((o1x-o0x).norm()))
print('Jtok absmax', np.abs(Jtok).max().round(4), 'Jtok[0,:4,0]', Jtok[0,:4,0].round(4), 'base4[0]', base4[0].round(1))
print('WT type', type(WT).__name__, 'z_in keys', [kk_ for kk_ in WT.keys() if 'z_in' in kk_][:4])
o = torch.as_tensor(X3n[k-1,:192], dtype=dtw); h = torch.as_tensor(X3n[k-1,192:960].reshape(4,192), dtype=dtw)
dtk = dts[k]; hf = h*torch.exp(-dtk*torch.exp(WT['core.flow.a'])); ctx = ctx_from(o, hf, dtk); z = zpost_mean(ctx, EtW[k], UtW[k])
idx_m = np.arange(max(j0of[k], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
_, oa = core_step(hf, EtW[k], UtW[k], z, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
_, ob = core_step(hf, EtW[k], UtW[k], z+2.0, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
print('z+2 dout', float((ob-oa).norm()), 'oa vs one_step', float((oa-o0x).norm()))
```