Adding the fitted u-token correction (norm 0.06, about 0.5% of |u|) to my local edited model moves its injection coefficients from (−8.2, +5.1) to (+22.4, −4.7), close to the tool's (+29.3, −4.3). The edited-model mismatch with the tool falls from 3.8% to 0.6%. So the tool's wrong-signed edit comes from my decoders being hypersensitive to a tiny difference in how the action token is processed. A 0.5% change in u flips the decoded insulin residual from +1.8 to about −5.

I had assumed "z exact implies u exact", but that only holds if z_post responds to this particular direction. If z_post is insensitive to the fitted correction, my action encoding itself may simply be slightly wrong everywhere. That would bring back encoding details such as the bin convention, which I can now test precisely.

So I'll check z's response to that correction. Then I'll score the four bin-convention combinations (left/right for rate × dose) by the single-step error from the tool state at six drug bundles.
- If z barely moves and one combination gives errors near 1e-6, I've found the true encoding and local and tool will agree.
- If not, I'll design the edit to be robust to this direction.

```python
dj = prep(3, Xo5)
globals()['WCUR']=W64
z_a = msx(dj)[2]; dd = dict(dj); dd['u'] = dj['u'] + torch.tensor(SOLU3); z_b = msx(dd)[2]
print('z rel change from SOLU3 in all u uses:', float((z_b-z_a).norm()/z_a.norm()), ' state err vs tool with u+SOLU3 everywhere:', np.linalg.norm(Xo5[idx[3]]-st_vec(msx(dd)))/np.linalg.norm(Xo5[idx[3]]))
def mk_fb2(rate_side, dose_side):
    def f(c, attr, x):
        e_ = np.array(NUMT[(c,attr)]['fine_bin_edges'])
        side = rate_side if attr=='rate' else (dose_side if attr=='dose' else 'left')
        return 1 + int(np.sum(e_ < x)) if side=='left' else 1 + int(np.sum(e_ <= x))
    return f
for rs in ['left','right']:
    for ds in ['left','right']:
        globals()['fb_'] = mk_fb2(rs, ds); errs=[]
        for j in [3,5,7,9,11,13]:
            dj = prep(j, Xo5); rj = st_vec(msx(dj)); errs.append(np.linalg.norm(Xo5[idx[j]]-rj)/np.linalg.norm(rj))
        print('rate', rs, 'dose', ds, np.round(errs,7))
globals()['fb_'] = fb_orig
```