head_run takes 0.2 s per dataset, so cross-fitted offline costs are cheap. But its output with the original weights does not match base_out[0] (max difference 9.17 in predicted mean, 1.0 in NLL). That could be a few rows or a systematic difference such as a different z seed. Before relying on either as the baseline, I'll look at the size and spread of the differences and repeat the run with another seed.

```python
pm_o, nll_o = np.asarray(o[0]), np.asarray(o[1])
pm_b, nll_b = np.asarray(base_out[0][0]), np.asarray(base_out[0][1])
d_pm = np.abs(pm_o-pm_b); d_nll = nll_o-nll_b
print('pm |diff| mean %.3f median %.3f frac>0.5 %.3f' % (d_pm.mean(), np.median(d_pm), (d_pm>0.5).mean()))
print('nll diff mean %.4f  |diff| mean %.4f' % (d_nll.mean(), np.abs(d_nll).mean()))
o2 = head_run(datasets[0][0], datasets[0][1], ORIG, seed=1)
print('seed1 vs seed0: pm |diff| mean %.3f, nll mean diff %.4f' % (np.abs(np.asarray(o2[0])-pm_o).mean(), (np.asarray(o2[1])-nll_o).mean()))
print('nll mean: head_run %.4f  base %.4f  rows file %.4f' % (nll_o.mean(), nll_b.mean(), datasets[0][1].nll.mean()))
```