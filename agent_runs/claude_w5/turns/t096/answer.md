Run 91 rules out the transforms: normalising or rescaling the atom vectors only makes things worse, so the atom embedding as documented looks right. I'm now at 91 runs and will stop the broad search soon. One cheap diagnostic remains. If the mismatch depends on position in the stay, being small for early bundles and growing later, then the state passed between steps contains something the stored states don't show. If it depends on the type of the previous bundle (drug versus measurement), the problem is in how the time flow interacts with drug bundles.

I'll print the error by bundle index, and by the previous bundle's type, using the best ctx found so far (no gap term). I'll also record what I've established in the notebook.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\n## Runs 70-91: rebuild check vs fwd_2 states\n- states x: [:192]=out, [192:960]=h slot-major (out==LN(mean h) exact, eps 1e-5), [960:992]=z post mean; rows align 1:1 with bundles_B.\n"
             "- embed tables: idx0 = padding (zero) for ekind/categorical/text/bins; untrained rows (~0.27 norm): action, discharged_to, route, status, text, kq 1-3 -> always 0 (kq=0 trained). unit 2-5 and rate_unit 2-3 used.\n"
             "- z_post([ctx,e,u]) order confirmed; GELU best everywhere; first 32 = mean. Numeric order [zv,hv,zd,hd,zr,hr] confirmed.\n"
             "- BUT z mismatch rel err ~0.6 (corr per dim 0.65-0.93) with ctx = out + LN(mean h_flow) (no gap). gap term as documented makes it worse. Visit-only bundles err 0.19. Not resolved.\n")
errk = defaultdict(list); errp = defaultdict(list)
for s_ in list(bundles_B.keys())[:30]:
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
    for k_ in range(1, len(bl_)):
        atoms = bl_[k_][1]
        wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
        if aa or not wa: continue
        prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
        dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
        zh = zfun((out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm'))[None], E_(wa)[None])[0][0]; zt = Xt[r0_+k_, 960:]
        er = float(torch.norm(zh-zt)/torch.norm(zt))
        errk[min(k_, 12)].append(er)
        pt = 'drug' if any(a['code'] in ACTION_CODES for a in bl_[k_-1][1]) else ('visit' if all(a.get('value') is None for a in bl_[k_-1][1]) else 'meas')
        errp[pt].append(er)
print('by k:', {k_: (len(v_), round(float(np.median(v_)),3)) for k_, v_ in sorted(errk.items())})
print('by prev type:', {k_: (len(v_), round(float(np.median(v_)),3)) for k_, v_ in errp.items()})
s1 = [s_ for s_ in bundles_B if any(set(a['code'] for a in b_[1]) == {'hospital_admission','icu_admission'} for b_ in bundles_B[s_])][:1]
if s1: print([(round(b_[0],3), [a['code'] for a in b_[1]]) for b_ in bundles_B[s1[0]][:4]])
```