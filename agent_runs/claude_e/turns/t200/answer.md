`gated3` returns short keys: `W0`, `b0`, `W2`, `TR`. The saved edit changes 4 units of the value head: their 4 biases, 4×192 input weights (up to 143 in size) and 4×17 output weights. The transition table is saved but identical to the original.

I'll map the short keys to tensor names and compare `gated3(-1.9,3)` with the saved edit. Then I'll evaluate the saved edit on sel1. If the saved edit gives a small map_arterial cost on sel1 but `gated3` now does not, the gate globals were changed after saving. If both explode on sel1, the problem is in my sel1 pipeline instead.

```python
KMAP = {'W0':'attr_heads.value_bin.0.weight','b0':'attr_heads.value_bin.0.bias','W2':'attr_heads.value_bin.2.weight','TR':'embed_transition.value_bin.weight'}
typ = type(E_g3['W0'])
for k, n in KMAP.items():
    a_ = np.asarray(E_g3[k], np.float64)
    print(k, 'gated3 now vs saved maxdiff %.4g' % float(np.abs(a_ - SAVED[n]).max()), '| vs orig %.4g' % float(np.abs(a_ - np.asarray(load_original(n),np.float64)).max()))
Wd_saved = {k: (torch.as_tensor(SAVED[n]) if typ is torch.Tensor else SAVED[n]) for k, n in KMAP.items()}
o_s = head_run(X, RR, Wd_saved)
RRc['dn_s'] = np.asarray(o_s[1], float) - nz_; RRc['dpm_s'] = np.asarray(o_s[0],float) - np.asarray(o_z[0],float)
print('saved edit on sel1: overall dNLL %.4f' % RRc.dn_s.mean())
print(RRc.groupby('code').agg(n=('dn_s','size'), dn=('dn_s','mean'), dpm_sd=('dpm_s','std')).round(4).to_string())
Wd_orig = {k: (torch.as_tensor(np.asarray(load_original(n),np.float64)) if typ is torch.Tensor else np.asarray(load_original(n),np.float64)) for k, n in KMAP.items()}
o_o = head_run(X, RR, Wd_orig)
print('orig vs gated3(0,0) on sel1: dNLL %.5f, max|dpm| %.4f' % ((np.asarray(o_o[1])-nz_).mean(), np.abs(np.asarray(o_o[0])-np.asarray(o_z[0])).max()))
```