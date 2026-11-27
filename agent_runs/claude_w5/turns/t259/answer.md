The decomposition settles it. The tool's edit effect at the first drug bundle lies almost entirely in my two injection directions (residual 1.2%). Its coefficients are (+29.3, −4.3) against local (−8.2, +5.1), so from identical input states the tool decodes the insulin residual as about −6.4 u/h where I decode +1.8. For that, n3 must differ by about 2% along my decoder direction. The final-state error looks small (1e-3) only because the state is dominated by the carried-over component of norm about 1800.

So my local prelude at drug bundles is genuinely off by a few percent, and the edit amplifies this. Next I'll split the j3 single-step error for the original model into out, each h slot, and z.
- If z, which depends directly on u, matches to about 1e-6, the action encoding u is right and the discrepancy is inside the prelude or blocks.
- If z is off too, u is mis-encoded, and I'll search over drug-atom encoding variants such as units, z-score clipping and dose features.

```python
dv = Xo5[idx[3]] - s0
parts = {'out': slice(0,192), 'h0': slice(192,384), 'h1': slice(384,576), 'h2': slice(576,768), 'h3': slice(768,960), 'z': slice(960,992)}
for k,sl in parts.items():
    print(k, 'abs err', np.linalg.norm(dv[sl]).round(5), 'rel', (np.linalg.norm(dv[sl])/np.linalg.norm(s0[sl])).round(6))
# same at a reading bundle j=4 and drug bundles j=5,7 (single step from tool state)
for j in [4,5,7,9]:
    dj = prep(j, Xo5); globals()['WCUR']=W64; rj = st_vec(mystep(dj)); dd = Xo5[idx[j]] - rj
    print(j, B[j][0], [a['code']+str(a.get('rate','')) for a in B[j][1]], {k: (np.linalg.norm(dd[sl])/np.linalg.norm(rj[sl])).round(6) for k,sl in parts.items()})
```