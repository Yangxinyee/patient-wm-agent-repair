Neither memory convention fits. Dropping the current e from memory makes the error about 50× worse at every drug bundle. Dropping the e token only matters at j3, where it makes things worse; at the later drug bundles the slots ignore e entirely. So memory and e-token handling are right, and `msx` reproduces `mystep` exactly.

The remaining suspect is still u. z is only weakly sensitive to u, so a u error lying in directions z_post largely ignores could leave z exact while moving h, since the prelude reads u heavily at drug bundles. I'll test this directly at j3. I'll compute the finite-difference Jacobian of the step's output state with respect to u, then solve for the δu that best explains the tool-minus-local difference.
- If a small δu (about 0.5% of |u|) cuts the error to near 1e-6 while z stays matched, u is mis-encoded. Comparing δu with candidate atom-feature changes will then show which feature is wrong.
- If δu can't explain it, the cause is elsewhere in the drug-bundle step.

```python
d3 = prep(3, Xo5); base = st_vec(msx(d3)); target = Xo5[idx[3]]
eps_ = 1e-4; Ju = np.zeros((992,192))
for i in range(192):
    dd = dict(d3); uu = d3['u'].clone(); uu[i] += eps_; dd['u'] = uu
    Ju[:,i] = (st_vec(msx(dd)) - base)/eps_
dlt = target - base
du, *_ = np.linalg.lstsq(Ju, dlt, rcond=None)
res = dlt - Ju@du
print('rel err before', np.linalg.norm(dlt)/np.linalg.norm(target), 'after u-fit', np.linalg.norm(res)/np.linalg.norm(target))
print('|du|', np.linalg.norm(du), '|u|', float(d3['u'].norm()), 'z part of Ju@du', np.linalg.norm((Ju@du)[960:]))
# verify nonlinearly
dd = dict(d3); dd['u'] = d3['u'] + torch.tensor(du); print('nonlinear check err', np.linalg.norm(st_vec(msx(dd))-target)/np.linalg.norm(target))
DU3 = du.copy()
```